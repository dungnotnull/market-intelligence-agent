"""
MarketIntelligenceOrchestrator: core decision loop for the market intelligence agent.
E2E pipeline: parse query -> collect data -> rank -> apply frameworks -> forecast -> report.
Includes cross-agent integration, Prometheus metrics, and benchmark instrumentation.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MarketIntelligenceOrchestrator:
    def __init__(self, config: dict = None):
        self._config = config or {}
        self._memory = None
        self._llm = None
        self._hf = None
        self._collector = None
        self._framework_analyzer = None
        self._report_generator = None
        self._trend_forecaster = None
        self._knowledge_updater = None
        self._scheduler = None
        self._prometheus_exporter = None
        self._benchmark_instrument = None
        self._citation_feed_client = None

    # ── Lazy initialization ─────────────────────────────────────────────────

    def _get_memory(self):
        if self._memory is None:
            from agent.memory.memory_manager import MemoryManager
            db_path = self._config.get("memory", {}).get("db_path", "data/market_intelligence.db")
            self._memory = MemoryManager(db_path=db_path)
        return self._memory

    def _get_llm(self):
        if self._llm is None:
            from tools.llm_client import UnifiedLLMClient
            self._llm = UnifiedLLMClient(memory_manager=self._get_memory())
        return self._llm

    def _get_hf(self):
        if self._hf is None:
            from tools.hf_model_manager import get_instance
            self._hf = get_instance()
        return self._hf

    def _get_collector(self):
        if self._collector is None:
            from agent.modules.data_collector import DataCollector
            timeout = self._config.get("data_collection", {}).get("timeout_seconds", 20)
            self._collector = DataCollector(timeout=timeout, config=self._config.get("data_collection", {}))
        return self._collector

    def _get_framework_analyzer(self):
        if self._framework_analyzer is None:
            from agent.modules.framework_analyzer import FrameworkAnalyzer
            self._framework_analyzer = FrameworkAnalyzer(
                llm_client=self._get_llm(), hf_manager=self._get_hf()
            )
        return self._framework_analyzer

    def _get_report_generator(self):
        if self._report_generator is None:
            from agent.modules.report_generator import ReportGenerator
            output_dir = self._config.get("output", {}).get("dir", "output")
            self._report_generator = ReportGenerator(
                llm_client=self._get_llm(), hf_manager=self._get_hf(), output_dir=output_dir
            )
        return self._report_generator

    def _get_trend_forecaster(self):
        if self._trend_forecaster is None:
            from agent.modules.trend_forecaster import TrendForecaster
            self._trend_forecaster = TrendForecaster(llm_client=self._get_llm())
        return self._trend_forecaster

    def _get_knowledge_updater(self):
        if self._knowledge_updater is None:
            from tools.knowledge_updater import KnowledgeUpdater
            self._knowledge_updater = KnowledgeUpdater(memory_manager=self._get_memory())
        return self._knowledge_updater

    def _get_prometheus_exporter(self):
        if self._prometheus_exporter is None:
            from agent.cross_agent.prometheus_exporter import get_exporter
            self._prometheus_exporter = get_exporter(memory_manager=self._get_memory())
        return self._prometheus_exporter

    def _get_benchmark_instrument(self):
        if self._benchmark_instrument is None:
            from agent.cross_agent.benchmark_instrument import get_instrument
            self._benchmark_instrument = get_instrument(memory_manager=self._get_memory())
        return self._benchmark_instrument

    def _get_citation_feed_client(self):
        if self._citation_feed_client is None:
            from agent.cross_agent.citation_feed import CitationFeedClient
            self._citation_feed_client = CitationFeedClient()
        return self._citation_feed_client

    # ── Main pipeline ───────────────────────────────────────────────────────

    async def analyze(
        self,
        query: str,
        sector: str = "",
        geo_scope: str = "global",
        depth: str = "standard",
        frameworks: list[str] = None,
        date_range: tuple[int, int] = None,
    ) -> dict:
        """Run the full market intelligence pipeline."""
        start_time = datetime.utcnow()
        start_mono = time.monotonic()
        logger.info("Starting analysis: query='%s' sector='%s' geo='%s'", query, sector, geo_scope)

        if not sector:
            sector = self._infer_sector(query)

        # Step 1: Collect data
        collector = self._get_collector()
        data_points = await collector.collect(
            query=query, sector=sector, geo_scope=geo_scope, date_range=date_range
        )
        logger.info("Collected %d data points", len(data_points))

        # Step 2: Semantic ranking (if HF available)
        top_points = await self._rank_data_points(query, data_points)

        # Step 3: Framework analysis
        framework_analyzer = self._get_framework_analyzer()
        framework_report = await framework_analyzer.analyze(
            query=query,
            evidence=top_points,
            sector=sector,
            geo_scope=geo_scope,
            requested_frameworks=frameworks,
        )
        logger.info("Framework analysis complete: %s", [fr.framework for fr in framework_report.frameworks])

        # Step 4: Trend forecasting
        forecaster = self._get_trend_forecaster()
        forecast_result = await forecaster.forecast(data_points)
        logger.info("Forecast complete: method=%s", forecast_result.method if forecast_result else "none")

        # Step 5: Report generation
        report_gen = self._get_report_generator()
        report_md, metadata = await report_gen.generate(
            query=query,
            sector=sector,
            geo_scope=geo_scope,
            framework_report=framework_report,
            forecast_result=forecast_result,
            data_points=data_points,
        )

        # Step 6: Persist to memory
        memory = self._get_memory()
        from agent.memory.memory_manager import AnalysisRecord
        record = AnalysisRecord(
            query=query,
            sector=sector,
            geo_scope=geo_scope,
            frameworks_used=[fr.framework for fr in framework_report.frameworks],
            sources_count=len(data_points),
            report_path=metadata.report_path,
            confidence_avg=metadata.confidence_avg,
            tokens_used=0,
            cost_usd=0.0,
        )
        analysis_id = memory.save_analysis(record)
        memory.save_data_sources(analysis_id, [
            {"source_name": dp.source_name, "url": dp.url, "title": dp.title,
             "snippet": dp.snippet, "confidence": dp.confidence, "data_type": dp.data_type}
            for dp in data_points
        ])

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        elapsed_mono = time.monotonic() - start_mono

        # Step 7: Record Prometheus metrics
        try:
            exporter = self._get_prometheus_exporter()
            exporter.record_analysis_complete(
                sector=sector,
                elapsed_seconds=elapsed_mono,
                sources_count=len(data_points),
                frameworks=[fr.framework for fr in framework_report.frameworks],
                confidence_avg=metadata.confidence_avg,
                word_count=metadata.word_count,
                quality_gates_passed=metadata.quality_gates_passed,
                quality_gates_total=metadata.quality_gates_total,
                forecast_method=metadata.forecast_method,
                cost_usd=0.0,
            )
        except Exception as e:
            logger.debug("Prometheus recording failed: %s", e)

        # Step 8: Feed citations to academic-research-enhanced (cross-agent)
        try:
            citation_client = self._get_citation_feed_client()
            if await citation_client.check_health():
                await citation_client.feed_report_citations(data_points)
                logger.info("Citations fed to academic-research-enhanced")
        except Exception as e:
            logger.debug("Cross-agent citation feed skipped: %s", e)

        logger.info("Analysis complete in %.1fs — %d words, %d/%d quality gates",
                    elapsed, metadata.word_count, metadata.quality_gates_passed, metadata.quality_gates_total)

        return {
            "status": "success",
            "query": query,
            "sector": sector,
            "geo_scope": geo_scope,
            "frameworks_applied": [fr.framework for fr in framework_report.frameworks],
            "sources_count": len(data_points),
            "unique_source_domains": metadata.unique_source_domains,
            "confidence_avg": metadata.confidence_avg,
            "word_count": metadata.word_count,
            "quality_gates": f"{metadata.quality_gates_passed}/{metadata.quality_gates_total}",
            "forecast_method": metadata.forecast_method,
            "forecast_cagr": forecast_result.cagr_projected if forecast_result else None,
            "report_path": metadata.report_path,
            "elapsed_seconds": round(elapsed, 1),
            "analysis_id": analysis_id,
            "report_preview": report_md[:800] + "..." if len(report_md) > 800 else report_md,
        }

    async def _rank_data_points(self, query: str, points: list) -> list:
        if not points:
            return []
        try:
            hf = self._get_hf()
            texts = [f"{p.title} {p.snippet[:200]}" for p in points]
            query_vec = hf.encode([query], model_key="bge_large")[0]
            doc_vecs = hf.encode(texts, model_key="bge_large")

            import numpy as np
            scores = doc_vecs @ query_vec
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranked = [points[i] for i in ranked_idx[:20]]

            passages = [f"{p.title}: {p.snippet[:300]}" for p in ranked]
            reranked = hf.rerank(query, passages, top_k=min(12, len(passages)))
            top_points = [ranked[i] for i, _ in reranked]
            return top_points
        except Exception as e:
            logger.warning("Ranking failed: %s — using first 12 points", e)
            return points[:12]

    # ── Knowledge update ─────────────────────────────────────────────────────

    async def update_knowledge(self) -> dict:
        updater = self._get_knowledge_updater()
        start = time.monotonic()
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, updater.run_update)
        elapsed = time.monotonic() - start

        try:
            exporter = self._get_prometheus_exporter()
            exporter.record_knowledge_update(count, elapsed)
        except Exception:
            pass

        return {"papers_added": count, "updated_at": datetime.utcnow().isoformat()}

    def start_scheduled_updates(self):
        updater = self._get_knowledge_updater()
        self._scheduler = updater.start_scheduled()

    # ── Cost reporting ───────────────────────────────────────────────────────

    def get_cost_report(self, days: int = 30) -> dict:
        return self._get_memory().get_cost_summary(days=days)

    def get_stats(self) -> dict:
        return self._get_memory().get_stats()

    # ── Prometheus metrics (enhanced for dockprom-enhanced) ──────────────────

    def get_prometheus_metrics(self) -> str:
        exporter = self._get_prometheus_exporter()
        return exporter.generate_metrics()

    # ── Benchmark instrumentation ────────────────────────────────────────────

    def get_benchmark_instrument(self):
        return self._get_benchmark_instrument()

    def get_citation_feed(self):
        return self._get_citation_feed_client()

    @staticmethod
    def _infer_sector(query: str) -> str:
        q = query.lower()
        sectors = {
            "ev": ["electric vehicle", "ev battery", "lithium"],
            "energy": ["energy", "renewable", "solar", "wind", "oil"],
            "agriculture": ["agriculture", "food", "farming"],
            "fintech": ["fintech", "banking", "finance", "payment"],
            "health": ["health", "medical", "pharma", "biotech"],
            "tech": ["saas", "software", "ai", "cloud", "tech"],
            "e-commerce": ["e-commerce", "retail", "ecommerce"],
        }
        for sector, keywords in sectors.items():
            if any(kw in q for kw in keywords):
                return sector
        words = query.split()[:3]
        return " ".join(words) if words else "general"
