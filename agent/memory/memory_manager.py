"""
Persistent memory manager for the Market Intelligence Agent.
SQLite WAL mode with 5 tables: analyses, data_sources, frameworks,
llm_cost_log, knowledge_hashes.
"""

import json
import sqlite3
import threading
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class AnalysisRecord:
    query: str
    sector: str
    geo_scope: str
    frameworks_used: list
    sources_count: int
    report_path: str
    confidence_avg: float
    tokens_used: int
    cost_usd: float
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class MemoryManager:
    def __init__(self, db_path: str = "data/market_intelligence.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    sector TEXT,
                    geo_scope TEXT,
                    frameworks_used TEXT,
                    sources_count INTEGER DEFAULT 0,
                    report_path TEXT,
                    confidence_avg REAL DEFAULT 0.0,
                    tokens_used INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_analyses_sector ON analyses(sector);
                CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);

                CREATE TABLE IF NOT EXISTS data_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    source_name TEXT,
                    url TEXT,
                    title TEXT,
                    snippet TEXT,
                    confidence REAL,
                    data_type TEXT,
                    retrieved_at TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
                );

                CREATE TABLE IF NOT EXISTS frameworks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    framework_name TEXT,
                    framework_json TEXT,
                    confidence_avg REAL,
                    created_at TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
                );

                CREATE TABLE IF NOT EXISTS llm_cost_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT,
                    model TEXT,
                    task TEXT,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS knowledge_hashes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_hash TEXT UNIQUE NOT NULL,
                    title TEXT,
                    source TEXT,
                    added_at TEXT
                );
            """)
            conn.commit()
            conn.close()

    # ── Analyses ──────────────────────────────────────────────────────────────

    def save_analysis(self, record: AnalysisRecord) -> int:
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                """INSERT INTO analyses
                   (query, sector, geo_scope, frameworks_used, sources_count,
                    report_path, confidence_avg, tokens_used, cost_usd, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.query,
                    record.sector,
                    record.geo_scope,
                    json.dumps(record.frameworks_used),
                    record.sources_count,
                    record.report_path,
                    record.confidence_avg,
                    record.tokens_used,
                    record.cost_usd,
                    record.created_at,
                ),
            )
            conn.commit()
            analysis_id = cur.lastrowid
            conn.close()
            return analysis_id

    def get_recent_analyses(self, limit: int = 10) -> list[dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

    def get_analysis_by_sector(self, sector: str) -> list[dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM analyses WHERE sector LIKE ? ORDER BY created_at DESC LIMIT 20",
                (f"%{sector}%",),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]

    # ── Data Sources ──────────────────────────────────────────────────────────

    def save_data_sources(self, analysis_id: int, sources: list[dict]):
        with self._lock:
            conn = self._connect()
            for s in sources:
                conn.execute(
                    """INSERT INTO data_sources
                       (analysis_id, source_name, url, title, snippet,
                        confidence, data_type, retrieved_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        analysis_id,
                        s.get("source_name", ""),
                        s.get("url", ""),
                        s.get("title", ""),
                        s.get("snippet", "")[:2000],
                        s.get("confidence", 0.5),
                        s.get("data_type", "qualitative"),
                        datetime.utcnow().isoformat(),
                    ),
                )
            conn.commit()
            conn.close()

    # ── Frameworks ────────────────────────────────────────────────────────────

    def save_framework(self, analysis_id: int, name: str, data: dict, confidence: float):
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO frameworks
                   (analysis_id, framework_name, framework_json, confidence_avg, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    analysis_id,
                    name,
                    json.dumps(data),
                    confidence,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            conn.close()

    # ── LLM Cost ──────────────────────────────────────────────────────────────

    def log_llm_cost(
        self,
        provider: str,
        model: str,
        task: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ):
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO llm_cost_log
                   (provider, model, task, tokens_in, tokens_out, cost_usd, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    provider,
                    model,
                    task,
                    tokens_in,
                    tokens_out,
                    cost_usd,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            conn.close()

    def get_cost_summary(self, days: int = 30) -> dict:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                """SELECT provider, model,
                          SUM(cost_usd) as total_cost,
                          SUM(tokens_in + tokens_out) as total_tokens,
                          COUNT(*) as call_count
                   FROM llm_cost_log WHERE created_at >= ?
                   GROUP BY provider, model ORDER BY total_cost DESC""",
                (since,),
            ).fetchall()
            total = conn.execute(
                "SELECT SUM(cost_usd) FROM llm_cost_log WHERE created_at >= ?",
                (since,),
            ).fetchone()[0]
            conn.close()
            return {
                "period_days": days,
                "total_cost_usd": round(total or 0.0, 4),
                "by_model": [dict(r) for r in rows],
            }

    # ── Knowledge Hashes ──────────────────────────────────────────────────────

    def is_known_paper(self, title: str, doi: str = "") -> bool:
        h = hashlib.sha256(f"{title}{doi}".encode()).hexdigest()
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT 1 FROM knowledge_hashes WHERE paper_hash = ?", (h,)
            ).fetchone()
            conn.close()
            return row is not None

    def mark_paper_known(self, title: str, source: str, doi: str = ""):
        h = hashlib.sha256(f"{title}{doi}".encode()).hexdigest()
        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT OR IGNORE INTO knowledge_hashes
                   (paper_hash, title, source, added_at) VALUES (?, ?, ?, ?)""",
                (h, title[:500], source, datetime.utcnow().isoformat()),
            )
            conn.commit()
            conn.close()

    def get_known_paper_hashes(self) -> set[str]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute("SELECT paper_hash FROM knowledge_hashes").fetchall()
            conn.close()
            return {r[0] for r in rows}

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            conn = self._connect()
            analyses_count = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            sources_count = conn.execute("SELECT COUNT(*) FROM data_sources").fetchone()[0]
            papers_count = conn.execute("SELECT COUNT(*) FROM knowledge_hashes").fetchone()[0]
            avg_conf = conn.execute(
                "SELECT AVG(confidence_avg) FROM analyses WHERE confidence_avg > 0"
            ).fetchone()[0]
            conn.close()
            return {
                "total_analyses": analyses_count,
                "total_sources_collected": sources_count,
                "knowledge_papers": papers_count,
                "avg_confidence": round(avg_conf or 0.0, 3),
            }
