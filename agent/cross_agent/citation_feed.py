"""
CitationFeedClient: feeds generated market intelligence reports to the
academic-research-enhanced agent (folder 18) citation database via REST API.
Supports batch submission, deduplication, and retry with exponential backoff.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CitationEntry:
    """Structured citation record for the academic-research-enhanced database."""
    title: str
    authors: str
    year: int
    source_name: str
    url: str
    abstract: str
    citation_type: str  # "statistical" | "academic" | "qualitative" | "regulatory"
    confidence: float
    sector: str = ""
    doi: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "source_name": self.source_name,
            "url": self.url,
            "abstract": self.abstract[:500],
            "citation_type": self.citation_type,
            "confidence": self.confidence,
            "sector": self.sector,
            "doi": self.doi,
            "tags": self.tags,
            "metadata": self.metadata,
            "fingerprint": self.fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.title}{self.url}".encode()).hexdigest()


class CitationFeedClient:
    """Client for pushing citations to academic-research-enhanced citation database."""

    def __init__(self, base_url: str = None, api_key: str = None):
        self._base_url = base_url or os.getenv(
            "ACADEMIC_RESEARCH_ENHANCED_URL",
            "http://localhost:8018"
        )
        self._api_key = api_key or os.getenv("ACADEMIC_RESEARCH_API_KEY", "")
        self._timeout = 15
        self._max_retries = 3
        self._backoff_base = 1.0

    async def submit_citations(self, citations: list[CitationEntry]) -> dict:
        """Submit a batch of citation entries to the citation database."""
        if not citations:
            return {"submitted": 0, "duplicates": 0, "errors": 0}

        payload = {
            "source_agent": "market-intelligence-agent",
            "submitted_at": datetime.utcnow().isoformat(),
            "citations": [c.to_dict() for c in citations],
        }
        url = f"{self._base_url}/api/v1/citations/batch"
        return await self._post_with_retry(url, payload)

    async def submit_single(self, citation: CitationEntry) -> dict:
        """Submit a single citation entry."""
        url = f"{self._base_url}/api/v1/citations"
        payload = {
            "source_agent": "market-intelligence-agent",
            "submitted_at": datetime.utcnow().isoformat(),
            **citation.to_dict(),
        }
        return await self._post_with_retry(url, payload)

    async def check_health(self) -> bool:
        """Check if the academic-research-enhanced service is reachable."""
        url = f"{self._base_url}/health"
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._get_json, url)
            return data is not None and data.get("status") == "ok"
        except Exception:
            return False

    async def get_citation_count(self, sector: str = "") -> Optional[int]:
        """Get the count of citations in the database, optionally filtered by sector."""
        url = f"{self._base_url}/api/v1/citations/count"
        if sector:
            url += f"?sector={sector}"
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._get_json, url)
            if data and "count" in data:
                return data["count"]
        except Exception:
            pass
        return None

    def convert_data_points_to_citations(self, data_points: list) -> list[CitationEntry]:
        """Convert DataPoint objects from DataCollector into CitationEntry objects."""
        citations = []
        for dp in data_points:
            year = datetime.utcnow().year
            if dp.date and len(dp.date) >= 4:
                try:
                    year = int(dp.date[:4])
                except ValueError:
                    pass
            citations.append(CitationEntry(
                title=dp.title,
                authors=dp.metadata.get("authors", dp.source_name),
                year=year,
                source_name=dp.source_name,
                url=dp.url,
                abstract=dp.snippet[:500],
                citation_type=dp.data_type,
                confidence=dp.confidence,
                sector=dp.sector,
                doi=dp.metadata.get("doi", ""),
                tags=[dp.data_type, dp.sector] if dp.sector else [dp.data_type],
                metadata={"numeric_value": dp.numeric_value, "numeric_unit": dp.numeric_unit}
                if dp.numeric_value is not None else {},
            ))
        return citations

    async def feed_report_citations(self, data_points: list) -> dict:
        """Convenience method: convert DataPoints and submit them as citations."""
        citations = self.convert_data_points_to_citations(data_points)
        return await self.submit_citations(citations)

    # ── Internal ────────────────────────────────────────────────────────────

    async def _post_with_retry(self, url: str, payload: dict) -> dict:
        last_err = None
        for attempt in range(self._max_retries):
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._post_json, url, payload
                )
                return result
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    delay = self._backoff_base * (2 ** attempt)
                    logger.warning(
                        "Citation feed attempt %d failed: %s — retrying in %.1fs",
                        attempt + 1, e, delay
                    )
                    await asyncio.sleep(delay)
        logger.error("Citation feed failed after %d attempts: %s", self._max_retries, last_err)
        return {"submitted": 0, "duplicates": 0, "errors": len(payload.get("citations", [payload]))}

    def _post_json(self, url: str, payload: dict) -> dict:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MarketIntelligenceAgent/1.0",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug("POST to %s failed: %s", url, e)
            raise

    def _get_json(self, url: str) -> Optional[dict]:
        import urllib.request
        headers = {"User-Agent": "MarketIntelligenceAgent/1.0"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
