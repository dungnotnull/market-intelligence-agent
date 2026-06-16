"""
DataCollector: asynchronously crawls authoritative global market data sources.
Sources: World Bank API, IMF SDMX REST API, arXiv, Semantic Scholar, DuckDuckGo search, free news RSS.
Returns DataPoint objects with source metadata and confidence scores.
Rate limiting and retry logic per source API.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

WORLD_BANK_BASE = "https://api.worldbank.org/v2"
IMF_SDMX_BASE = "https://dataservices.imf.org/REST/SDMX_JSON.svc"

SECTOR_WB_INDICATORS = {
    "ev":          ["EG.ELC.ACCS.ZS", "EN.ATM.CO2E.KT"],
    "energy":      ["EG.USE.PCAP.KG.OE", "EG.ELC.ACCS.ZS"],
    "agriculture": ["AG.SRF.TOTL.K2", "NV.AGR.TOTL.ZS"],
    "finance":     ["FS.AST.DOMS.GD.ZS", "CM.MKT.TRAD.GD.ZS"],
    "health":      ["SH.XPD.CHEX.GD.ZS", "SP.DYN.LE00.IN"],
    "tech":        ["IT.NET.USER.ZS", "IT.CEL.SETS.P2"],
    "default":     ["NY.GDP.MKTP.CD", "NY.GDP.MKTP.KD.ZG"],
}

SECTOR_IMF_INDICATORS = {
    "ev":          ["NGDP_R", "NGDP_D"],
    "energy":      ["NGDP_R", "PCPI_IX"],
    "agriculture": ["NGDP_R", "TXG_FOB_USD"],
    "finance":     ["NGDP_D", "PCPI_IX"],
    "health":      ["NGDP_D", "PCPI_IX"],
    "tech":        ["NGDP_R", "PCPI_IX"],
    "default":     ["NGDP_R", "NGDP_D"],
}

IMF_DATABASES = {
    "WEO": "WEO",
    "IFS": "IFS",
    "BOP": "BOP",
    "DOT": "DOT",
}

# Per-source rate limiting configuration (requests per second)
SOURCE_RATE_LIMITS = {
    "world_bank": {"rps": 5.0, "burst": 10},
    "imf":        {"rps": 2.0, "burst": 5},
    "arxiv":      {"rps": 1.0, "burst": 3},
    "semantic_scholar": {"rps": 1.0, "burst": 3},
    "duckduckgo": {"rps": 2.0, "burst": 5},
}

# Per-source retry configuration
SOURCE_RETRY_CONFIG = {
    "world_bank":        {"max_retries": 3, "backoff_base": 1.0, "backoff_max": 8.0, "timeout": 20},
    "imf":               {"max_retries": 3, "backoff_base": 2.0, "backoff_max": 16.0, "timeout": 25},
    "arxiv":             {"max_retries": 2, "backoff_base": 2.0, "backoff_max": 12.0, "timeout": 30},
    "semantic_scholar":  {"max_retries": 3, "backoff_base": 2.0, "backoff_max": 16.0, "timeout": 20},
    "duckduckgo":        {"max_retries": 2, "backoff_base": 1.0, "backoff_max": 8.0, "timeout": 15},
}


@dataclass
class DataPoint:
    source_name: str
    url: str
    title: str
    snippet: str
    date: str
    confidence: float
    data_type: str  # "statistical" | "qualitative" | "regulatory" | "academic"
    sector: str = ""
    numeric_value: Optional[float] = None
    numeric_unit: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def source_domain(self) -> str:
        try:
            return urllib.parse.urlparse(self.url).netloc
        except Exception:
            return self.source_name


class RateLimiter:
    """Token-bucket rate limiter for per-source API calls."""

    def __init__(self, rps: float = 2.0, burst: int = 5):
        self._rps = rps
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rps)
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rps
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class RetryHandler:
    """Exponential backoff retry handler with per-source configuration."""

    def __init__(self, max_retries: int = 3, backoff_base: float = 1.0,
                 backoff_max: float = 8.0):
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

    async def execute(self, fn, *args, **kwargs):
        last_err = None
        for attempt in range(self._max_retries):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    delay = min(
                        self._backoff_base * (2 ** attempt),
                        self._backoff_max
                    )
                    jitter = delay * 0.1 * (hash(str(e)) % 10 / 10.0)
                    logger.warning(
                        "Attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1, self._max_retries, e, delay + jitter
                    )
                    await asyncio.sleep(delay + jitter)
        raise RuntimeError(
            f"All {self._max_retries} retries exhausted: {last_err}"
        )


class DataCollector:
    def __init__(self, timeout: int = 20, config: dict = None):
        self._timeout = timeout
        self._config = config or {}
        self._rate_limiters = {}
        self._retry_handlers = {}
        self._init_rate_limiters()
        self._init_retry_handlers()

    def _init_rate_limiters(self):
        for source, limits in SOURCE_RATE_LIMITS.items():
            self._rate_limiters[source] = RateLimiter(
                rps=limits["rps"], burst=limits["burst"]
            )

    def _init_retry_handlers(self):
        for source, retry_cfg in SOURCE_RETRY_CONFIG.items():
            self._retry_handlers[source] = RetryHandler(
                max_retries=retry_cfg["max_retries"],
                backoff_base=retry_cfg["backoff_base"],
                backoff_max=retry_cfg["backoff_max"],
            )

    def get_rate_limiter(self, source: str) -> RateLimiter:
        return self._rate_limiters.get(source, RateLimiter())

    def get_retry_handler(self, source: str) -> RetryHandler:
        return self._retry_handlers.get(source, RetryHandler())

    # ── Public API ──────────────────────────────────────────────────────────

    async def collect(
        self,
        query: str,
        sector: str = "",
        geo_scope: str = "global",
        date_range: tuple[int, int] = None,
    ) -> list[DataPoint]:
        if date_range is None:
            current_year = datetime.utcnow().year
            date_range = (current_year - 5, current_year)

        tasks = [
            self._collect_world_bank(query, sector, geo_scope, date_range),
            self._collect_imf(query, sector, geo_scope, date_range),
            self._collect_arxiv(query, sector),
            self._collect_semantic_scholar(query),
            self._collect_duckduckgo(query, sector, geo_scope),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_points: list[DataPoint] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Collection task failed: %s", r)
            elif isinstance(r, list):
                all_points.extend(r)

        all_points = self._deduplicate(all_points)
        all_points.sort(key=lambda p: p.confidence, reverse=True)

        unique_domains = {p.source_domain for p in all_points}
        if len(unique_domains) < 3:
            extra = await self._collect_duckduckgo(f"{query} market report {geo_scope}", sector, geo_scope)
            all_points.extend(extra)
            all_points = self._deduplicate(all_points)

        logger.info("Collected %d data points from %d sources for: %s", len(all_points), len({p.source_domain for p in all_points}), query)
        return all_points

    # ── World Bank ───────────────────────────────────────────────────────────

    async def _collect_world_bank(
        self, query: str, sector: str, geo_scope: str, date_range: tuple[int, int]
    ) -> list[DataPoint]:
        points = []
        sector_key = self._map_sector_to_wb(sector or query)
        indicators = SECTOR_WB_INDICATORS.get(sector_key, SECTOR_WB_INDICATORS["default"])
        country = self._map_geo_to_wb_code(geo_scope)
        start_year, end_year = date_range
        rate_limiter = self.get_rate_limiter("world_bank")
        retry_handler = self.get_retry_handler("world_bank")

        for indicator in indicators[:2]:
            url = (
                f"{WORLD_BANK_BASE}/country/{country}/indicator/{indicator}"
                f"?format=json&date={start_year}:{end_year}&mrv=5&per_page=50"
            )
            try:
                await rate_limiter.acquire()

                async def _fetch(url=url):
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self._fetch_json, url)

                data = await retry_handler.execute(_fetch)
                if not data or len(data) < 2:
                    continue
                entries = data[1] or []
                for entry in entries[:5]:
                    value = entry.get("value")
                    if value is None:
                        continue
                    year = entry.get("date", "")
                    country_name = entry.get("country", {}).get("value", country)
                    indicator_name = entry.get("indicator", {}).get("value", indicator)
                    points.append(DataPoint(
                        source_name="World Bank Open Data",
                        url=f"https://data.worldbank.org/indicator/{indicator}",
                        title=f"{indicator_name} — {country_name} ({year})",
                        snippet=f"{indicator_name}: {value:.2f} ({year}) for {country_name}",
                        date=f"{year}-01-01",
                        confidence=0.9,
                        data_type="statistical",
                        sector=sector,
                        numeric_value=float(value),
                        numeric_unit=indicator,
                    ))
            except Exception as e:
                logger.debug("World Bank API error for %s: %s", indicator, e)

        return points

    # ── IMF SDMX REST API ───────────────────────────────────────────────────

    async def _collect_imf(
        self, query: str, sector: str, geo_scope: str, date_range: tuple[int, int]
    ) -> list[DataPoint]:
        points = []
        sector_key = self._map_sector_to_wb(sector or query)
        imf_indicators = SECTOR_IMF_INDICATORS.get(sector_key, SECTOR_IMF_INDICATORS["default"])
        imf_country = self._map_geo_to_imf_code(geo_scope)
        start_year, end_year = date_range
        rate_limiter = self.get_rate_limiter("imf")
        retry_handler = self.get_retry_handler("imf")

        for indicator in imf_indicators[:2]:
            url = (
                f"{IMF_SDMX_BASE}/CompactData/{IMF_DATABASES['IFS']}"
                f"/{indicator}.{imf_country}?startPeriod={start_year}&endPeriod={end_year}"
            )
            try:
                await rate_limiter.acquire()

                async def _fetch(url=url):
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self._fetch_json, url)

                data = await retry_handler.execute(_fetch)
                if not data:
                    continue

                series_data = (
                    data.get("CompactData", {})
                    .get("DataSet", {})
                    .get("Series", {})
                )
                if not series_data:
                    continue

                obs_list = series_data.get("Obs", [])
                if not isinstance(obs_list, list):
                    obs_list = [obs_list] if obs_list else []

                indicator_name = series_data.get("@INDICATOR", indicator)
                country_name = series_data.get("@REF_AREA", imf_country)

                for obs in obs_list[:5]:
                    period = obs.get("@TIME_PERIOD", "")
                    value_str = obs.get("@OBS_VALUE", "")
                    if not value_str or not period:
                        continue
                    try:
                        value = float(value_str)
                    except (ValueError, TypeError):
                        continue
                    year = period[:4] if len(period) >= 4 else period
                    points.append(DataPoint(
                        source_name="IMF SDMX",
                        url=f"https://data.imf.org/regular.aspx?contentid=28",
                        title=f"IMF {indicator_name} — {country_name} ({period})",
                        snippet=f"IMF {indicator_name}: {value:.2f} ({period}) for {country_name}",
                        date=f"{year}-01-01" if len(year) == 4 else f"{period}",
                        confidence=0.85,
                        data_type="statistical",
                        sector=sector,
                        numeric_value=value,
                        numeric_unit=indicator,
                    ))
            except Exception as e:
                logger.debug("IMF SDMX API error for %s: %s", indicator, e)

        if not points:
            points = await self._collect_imf_fallback(query, sector, geo_scope, date_range)

        return points

    async def _collect_imf_fallback(
        self, query: str, sector: str, geo_scope: str, date_range: tuple[int, int]
    ) -> list[DataPoint]:
        """Fallback: use IMF WEO data via simplified JSON endpoint."""
        points = []
        start_year, end_year = date_range
        url = (
            f"{IMF_SDMX_BASE}/CompactData/{IMF_DATABASES['WEO']}"
            f"/NGDP_R.{self._map_geo_to_imf_code(geo_scope)}?startPeriod={start_year}&endPeriod={end_year}"
        )
        rate_limiter = self.get_rate_limiter("imf")
        retry_handler = self.get_retry_handler("imf")
        try:
            await rate_limiter.acquire()

            async def _fetch():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._fetch_json, url)

            data = await retry_handler.execute(_fetch)
            if not data:
                return points
            series = data.get("CompactData", {}).get("DataSet", {}).get("Series", {})
            obs_list = series.get("Obs", [])
            if not isinstance(obs_list, list):
                obs_list = [obs_list] if obs_list else []
            for obs in obs_list[:5]:
                period = obs.get("@TIME_PERIOD", "")
                value_str = obs.get("@OBS_VALUE", "")
                if not value_str or not period:
                    continue
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    continue
                year = period[:4] if len(period) >= 4 else period
                points.append(DataPoint(
                    source_name="IMF WEO",
                    url="https://www.imf.org/en/Publications/WEO",
                    title=f"IMF WEO GDP — {geo_scope} ({period})",
                    snippet=f"IMF WEO Real GDP growth: {value:.2f}% ({period}) for {geo_scope}",
                    date=f"{year}-01-01",
                    confidence=0.8,
                    data_type="statistical",
                    sector=sector,
                    numeric_value=value,
                    numeric_unit="NGDP_R",
                ))
        except Exception as e:
            logger.debug("IMF WEO fallback error: %s", e)
        return points

    def _map_geo_to_imf_code(self, geo_scope: str) -> str:
        g = geo_scope.lower()
        codes = {
            "global": "W00",
            "world": "W00",
            "us": "111",
            "usa": "111",
            "china": "924",
            "eu": "163",
            "europe": "163",
            "southeast asia": "819",
            "asean": "819",
            "vietnam": "582",
            "india": "534",
            "japan": "158",
            "korea": "542",
            "germany": "134",
            "uk": "112",
            "brazil": "223",
        }
        for key, code in codes.items():
            if key in g:
                return code
        return "W00"

    # ── arXiv ────────────────────────────────────────────────────────────────

    async def _collect_arxiv(self, query: str, sector: str) -> list[DataPoint]:
        search_terms = f"{query} market analysis"
        params = urllib.parse.urlencode({
            "search_query": f"all:{search_terms}",
            "start": 0,
            "max_results": 10,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"https://export.arxiv.org/api/query?{params}"
        rate_limiter = self.get_rate_limiter("arxiv")
        retry_handler = self.get_retry_handler("arxiv")
        try:
            await rate_limiter.acquire()

            async def _fetch():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._fetch_text, url)

            xml_text = await retry_handler.execute(_fetch)
            return self._parse_arxiv_to_datapoints(xml_text, sector)
        except Exception as e:
            logger.debug("arXiv collection error: %s", e)
            return []

    def _parse_arxiv_to_datapoints(self, xml_text: str, sector: str) -> list[DataPoint]:
        points = []
        try:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns)[:8]:
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                id_el = entry.find("atom:id", ns)
                published_el = entry.find("atom:published", ns)
                if not title_el or not id_el:
                    continue
                title = title_el.text.strip().replace("\n", " ")
                abstract = (summary_el.text or "").strip()[:400]
                date_str = (published_el.text or "")[:10] if published_el is not None else ""
                points.append(DataPoint(
                    source_name="arXiv",
                    url=id_el.text.strip(),
                    title=title,
                    snippet=abstract,
                    date=date_str,
                    confidence=0.75,
                    data_type="academic",
                    sector=sector,
                ))
        except Exception as e:
            logger.debug("arXiv XML parse error: %s", e)
        return points

    # ── Semantic Scholar ─────────────────────────────────────────────────────

    async def _collect_semantic_scholar(self, query: str) -> list[DataPoint]:
        params = urllib.parse.urlencode({
            "query": f"{query} market analysis",
            "limit": 8,
            "fields": "title,year,abstract,authors,externalIds,citationCount",
        })
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        rate_limiter = self.get_rate_limiter("semantic_scholar")
        retry_handler = self.get_retry_handler("semantic_scholar")
        try:
            await rate_limiter.acquire()

            async def _fetch():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._fetch_json, url)

            data = await retry_handler.execute(_fetch)
            papers = data.get("data", []) if data else []
            points = []
            for p in papers[:6]:
                ext = p.get("externalIds", {})
                doi = ext.get("DOI", "")
                arxiv_id = ext.get("ArXiv", "")
                paper_url = (
                    f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                    else f"https://doi.org/{doi}" if doi
                    else ""
                )
                if not paper_url or not p.get("title"):
                    continue
                abstract = (p.get("abstract") or "")[:400]
                year = p.get("year") or datetime.utcnow().year
                citations = p.get("citationCount", 0)
                confidence = min(0.9, 0.6 + min(citations, 100) / 500.0)
                points.append(DataPoint(
                    source_name="Semantic Scholar",
                    url=paper_url,
                    title=p["title"],
                    snippet=abstract,
                    date=f"{year}-01-01",
                    confidence=confidence,
                    data_type="academic",
                    metadata={"citations": citations},
                ))
            return points
        except Exception as e:
            logger.debug("Semantic Scholar error: %s", e)
            return []

    # ── DuckDuckGo search ────────────────────────────────────────────────────

    async def _collect_duckduckgo(self, query: str, sector: str, geo_scope: str) -> list[DataPoint]:
        search_query = f"{query} {geo_scope} market report 2024 2025"
        rate_limiter = self.get_rate_limiter("duckduckgo")
        retry_handler = self.get_retry_handler("duckduckgo")
        try:
            await rate_limiter.acquire()

            async def _search():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._ddgs_search, search_query)

            results = await retry_handler.execute(_search)
            points = []
            for r in results[:8]:
                points.append(DataPoint(
                    source_name="DuckDuckGo News",
                    url=r.get("href", r.get("url", "")),
                    title=r.get("title", ""),
                    snippet=r.get("body", r.get("description", ""))[:400],
                    date=r.get("date", datetime.utcnow().strftime("%Y-%m-%d")),
                    confidence=0.6,
                    data_type="qualitative",
                    sector=sector,
                ))
            return points
        except ImportError:
            return self._fallback_market_data_points(query, sector, geo_scope)
        except Exception as e:
            logger.debug("DuckDuckGo search error: %s", e)
            return self._fallback_market_data_points(query, sector, geo_scope)

    def _ddgs_search(self, query: str) -> list[dict]:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=8))

    def _fallback_market_data_points(self, query: str, sector: str, geo_scope: str) -> list[DataPoint]:
        current_year = datetime.utcnow().year
        return [
            DataPoint(
                source_name="Statista (reference)",
                url="https://www.statista.com/",
                title=f"Market data reference: {query}",
                snippet=f"Statista provides industry statistics and market data for {sector or query} sector. "
                        f"Please check statista.com for the latest {geo_scope} market figures.",
                date=f"{current_year}-01-01",
                confidence=0.5,
                data_type="qualitative",
                sector=sector,
            ),
            DataPoint(
                source_name="IMF World Economic Outlook",
                url="https://www.imf.org/en/Publications/WEO",
                title=f"IMF economic outlook reference: {geo_scope}",
                snippet=f"The IMF World Economic Outlook provides macroeconomic forecasts for {geo_scope}. "
                        f"GDP growth projections and sector outlooks are regularly updated.",
                date=f"{current_year}-01-01",
                confidence=0.7,
                data_type="statistical",
                sector=sector,
            ),
        ]

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _fetch_json(self, url: str) -> Optional[dict]:
        req = urllib.request.Request(url, headers={"User-Agent": "MarketIntelligenceAgent/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug("JSON fetch error %s: %s", url, e)
            return None

    def _fetch_text(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "MarketIntelligenceAgent/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            logger.debug("Text fetch error %s: %s", url, e)
            return ""

    def _deduplicate(self, points: list[DataPoint]) -> list[DataPoint]:
        seen_urls = set()
        result = []
        for p in points:
            key = hashlib.sha256(p.url.encode()).hexdigest() if p.url else str(id(p))
            if key not in seen_urls:
                seen_urls.add(key)
                result.append(p)
        return result

    def _map_sector_to_wb(self, sector_or_query: str) -> str:
        s = sector_or_query.lower()
        mapping = {
            "ev": ["electric", "battery", "ev "],
            "energy": ["energy", "renewable", "solar", "oil"],
            "agriculture": ["agriculture", "food", "farm"],
            "finance": ["fintech", "banking", "finance", "financial"],
            "health": ["health", "medical", "pharma", "biotech"],
            "tech": ["technology", "software", "ai", "digital", "saas"],
        }
        for key, keywords in mapping.items():
            if any(kw in s for kw in keywords):
                return key
        return "default"

    def _map_geo_to_wb_code(self, geo_scope: str) -> str:
        g = geo_scope.lower()
        codes = {
            "global": "WLD",
            "world": "WLD",
            "us": "US",
            "usa": "US",
            "china": "CN",
            "eu": "EUU",
            "europe": "EUU",
            "southeast asia": "EAS",
            "asean": "EAS",
            "vietnam": "VN",
            "india": "IN",
            "japan": "JP",
            "korea": "KR",
        }
        for key, code in codes.items():
            if key in g:
                return code
        return "WLD"
