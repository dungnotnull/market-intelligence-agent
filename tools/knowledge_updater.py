"""
Knowledge updater for Market Intelligence Agent.
Crawls ArXiv (cs.IR, cs.AI, stat.AP, cs.LG, econ.GN) + Semantic Scholar
and appends new papers to SECOND-KNOWLEDGE-BRAIN.md.
APScheduler: weekly, Sunday 02:00.
"""

import hashlib
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ARXIV_CATEGORIES = ["cs.IR", "cs.AI", "stat.AP", "cs.LG", "econ.GN"]

ARXIV_QUERIES = [
    "market intelligence LLM retrieval",
    "business competitive analysis machine learning",
    "market research automation NLP",
    "time series forecasting economic",
    "SWOT analysis automation language model",
]

S2_QUERIES = [
    "market intelligence artificial intelligence",
    "competitive analysis retrieval augmented generation",
    "economic forecasting deep learning",
    "business strategy natural language processing",
    "market research text mining",
]

DOMAIN_KEYWORDS = [
    "market intelligence",
    "competitive analysis",
    "SWOT",
    "Porter",
    "PESTEL",
    "market forecasting",
    "business strategy",
    "retrieval augmented",
    "economic indicators",
    "industry analysis",
    "market size",
    "competitive intelligence",
    "sector analysis",
    "market research",
]

BRAIN_PATH = Path("SECOND-KNOWLEDGE-BRAIN.md")


@dataclass
class PaperEntry:
    title: str
    authors: str
    year: int
    venue: str
    url: str
    abstract: str
    relevance_score: float = 0.0


def _paper_hash(entry: PaperEntry) -> str:
    return hashlib.sha256(f"{entry.title}{entry.url}".encode()).hexdigest()


def _parse_arxiv_xml(xml_text: str) -> list[PaperEntry]:
    papers = []
    try:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)
            id_el = entry.find("atom:id", ns)
            authors = [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]
            if not title_el or not id_el:
                continue
            year = int(published_el.text[:4]) if published_el is not None else datetime.utcnow().year
            papers.append(PaperEntry(
                title=title_el.text.strip().replace("\n", " "),
                authors=", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                year=year,
                venue="arXiv",
                url=id_el.text.strip(),
                abstract=(summary_el.text or "").strip()[:500],
            ))
    except Exception as e:
        logger.warning("arXiv XML parse error: %s", e)
    return papers


def _fetch_arxiv(query: str, max_results: int = 20) -> list[PaperEntry]:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MarketIntelligenceAgent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return _parse_arxiv_xml(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning("arXiv fetch failed for '%s': %s", query, e)
        return []


def _fetch_semantic_scholar(query: str, max_results: int = 10) -> list[PaperEntry]:
    params = urllib.parse.urlencode({
        "query": query,
        "limit": max_results,
        "fields": "title,year,abstract,authors,externalIds,citationCount",
    })
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    try:
        import json
        req = urllib.request.Request(url, headers={"User-Agent": "MarketIntelligenceAgent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        papers = []
        for p in data.get("data", []):
            ext = p.get("externalIds", {})
            doi = ext.get("DOI", "")
            arxiv_id = ext.get("ArXiv", "")
            url_p = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (f"https://doi.org/{doi}" if doi else "")
            if not url_p:
                continue
            authors = [a.get("name", "") for a in p.get("authors", [])[:3]]
            papers.append(PaperEntry(
                title=p.get("title", ""),
                authors=", ".join(authors) + (" et al." if len(p.get("authors", [])) > 3 else ""),
                year=p.get("year") or datetime.utcnow().year,
                venue="Semantic Scholar",
                url=url_p,
                abstract=(p.get("abstract") or "")[:500],
            ))
        return papers
    except Exception as e:
        logger.warning("Semantic Scholar fetch failed for '%s': %s", query, e)
        return []


def _score_paper(paper: PaperEntry) -> float:
    now = datetime.utcnow()
    age_days = max(0, (now - datetime(paper.year, 6, 15)).days)
    recency = max(0.0, 1.0 - age_days / 365.0)

    text = f"{paper.title} {paper.abstract}".lower()
    kw_hits = sum(1 for kw in DOMAIN_KEYWORDS if kw.lower() in text)
    relevance = min(1.0, kw_hits / 5.0)

    return 0.6 * recency + 0.4 * relevance


def _deduplicate(papers: list[PaperEntry], known_hashes: set[str]) -> list[PaperEntry]:
    seen = set()
    result = []
    for p in papers:
        h = _paper_hash(p)
        if h not in known_hashes and h not in seen:
            seen.add(h)
            result.append(p)
    return result


def _append_to_brain(papers: list[PaperEntry]):
    if not papers:
        return
    brain_text = BRAIN_PATH.read_text(encoding="utf-8") if BRAIN_PATH.exists() else ""

    log_marker = "## Knowledge Update Log"
    table_marker = "## Key Research Papers"

    new_rows = []
    for p in papers:
        abstract_short = p.abstract[:120].replace("|", "/").replace("\n", " ")
        new_rows.append(
            f"| {p.title[:70]} | {p.authors[:40]} | {p.year} | {p.venue} | {p.url} | {abstract_short}... | Auto-crawled |"
        )

    log_entry = (
        f"\n| {datetime.utcnow().strftime('%Y-%m-%d')} | ArXiv + Semantic Scholar "
        f"| {len(papers)} | Auto-crawled weekly run — {len(papers)} new entries |"
    )

    if table_marker in brain_text:
        table_block_end = brain_text.find("\n---", brain_text.find(table_marker))
        if table_block_end == -1:
            table_block_end = len(brain_text)
        insert_at = brain_text.rfind("\n", brain_text.find(table_marker), table_block_end)
        new_brain = brain_text[:insert_at] + "\n" + "\n".join(new_rows) + brain_text[insert_at:]
    else:
        new_brain = brain_text + "\n\n## Key Research Papers (Auto-Crawled)\n\n" + "\n".join(new_rows)

    if log_marker in new_brain:
        log_section_end = new_brain.find("\n", new_brain.rfind("|", new_brain.find(log_marker)))
        if log_section_end == -1:
            log_section_end = len(new_brain)
        new_brain = new_brain[:log_section_end] + log_entry + new_brain[log_section_end:]
    else:
        new_brain += f"\n\n{log_marker}\n\n| Date | Source | Papers Added | Notes |\n|------|--------|-------------|-------|{log_entry}\n"

    BRAIN_PATH.write_text(new_brain, encoding="utf-8")
    logger.info("Appended %d papers to SECOND-KNOWLEDGE-BRAIN.md", len(papers))


class KnowledgeUpdater:
    def __init__(self, memory_manager=None):
        self._memory = memory_manager

    def run_update(self) -> int:
        known_hashes = self._memory.get_known_paper_hashes() if self._memory else set()

        all_papers: list[PaperEntry] = []
        for query in ARXIV_QUERIES:
            all_papers.extend(_fetch_arxiv(query, max_results=15))
        for query in S2_QUERIES:
            all_papers.extend(_fetch_semantic_scholar(query, max_results=8))

        for p in all_papers:
            p.relevance_score = _score_paper(p)
        all_papers.sort(key=lambda p: p.relevance_score, reverse=True)

        new_papers = _deduplicate(all_papers, known_hashes)
        top_papers = new_papers[:10]

        _append_to_brain(top_papers)

        if self._memory:
            for p in top_papers:
                self._memory.mark_paper_known(p.title, p.venue, p.url)

        logger.info("Knowledge update complete: %d new papers added", len(top_papers))
        return len(top_papers)

    def start_scheduled(self):
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            scheduler = BackgroundScheduler()
            scheduler.add_job(
                self.run_update,
                trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
                id="knowledge_update_weekly",
                replace_existing=True,
            )
            scheduler.start()
            logger.info("Weekly knowledge updater scheduled (Sunday 02:00)")
            return scheduler
        except ImportError:
            logger.warning("APScheduler not installed — scheduled updates disabled")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    updater = KnowledgeUpdater()
    count = updater.run_update()
    print(f"Added {count} new papers to SECOND-KNOWLEDGE-BRAIN.md")
