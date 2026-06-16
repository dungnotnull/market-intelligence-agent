"""
Automated tests for the Market Intelligence Agent.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / 'test.db')


@pytest.fixture
def memory_manager(tmp_db):
    from agent.memory.memory_manager import MemoryManager
    return MemoryManager(db_path=tmp_db)


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    from tools.llm_client import LLMResult
    async def complete(prompt, system='', max_tokens=1000, task='general'):
        return LLMResult(text='{"frameworks": ["SWOT"], "rationale": "test"}',
            provider='mock', model='mock-model', tokens_in=100, tokens_out=50,
            cost_usd=0.001, latency_ms=100.0)
    mock.complete = complete
    return mock


@pytest.fixture
def sample_data_points():
    from agent.modules.data_collector import DataPoint
    return [
        DataPoint(source_name='World Bank', url='https://data.worldbank.org/indicator/NY.GDP.MKTP.CD',
            title='GDP - Vietnam (2023)', snippet='GDP of Vietnam reached 430 billion USD in 2023.',
            date='2023-01-01', confidence=0.9, data_type='statistical', sector='fintech',
            numeric_value=430.0, numeric_unit='billion USD'),
        DataPoint(source_name='arXiv', url='https://arxiv.org/abs/2303.12345',
            title='AI-Driven Market Analysis', snippet='ML methods for market intelligence.',
            date='2023-03-15', confidence=0.75, data_type='academic', sector='fintech'),
        DataPoint(source_name='Semantic Scholar', url='https://doi.org/10.1234/test',
            title='Fintech Market Growth', snippet='Fintech market projected to grow at 15% CAGR.',
            date='2022-06-01', confidence=0.8, data_type='academic', sector='fintech'),
        DataPoint(source_name='DuckDuckGo News', url='https://example.com/fintech-news',
            title='Vietnam Fintech Regulation 2024', snippet='New fintech sandbox regulations approved.',
            date='2024-03-01', confidence=0.6, data_type='qualitative', sector='fintech'),
    ]


class TestDataCollector:
    def test_map_sector_to_wb(self):
        from agent.modules.data_collector import DataCollector
        dc = DataCollector()
        assert dc._map_sector_to_wb('electric vehicle battery') == 'ev'
        assert dc._map_sector_to_wb('banking fintech') == 'finance'
        assert dc._map_sector_to_wb('random unknown sector') == 'default'

    def test_map_geo_to_imf_code(self):
        from agent.modules.data_collector import DataCollector
        dc = DataCollector()
        assert dc._map_geo_to_imf_code('global') == 'W00'
        assert dc._map_geo_to_imf_code('Vietnam') == '582'
        assert dc._map_geo_to_imf_code('US') == '111'

    def test_deduplicate(self):
        from agent.modules.data_collector import DataCollector, DataPoint
        dc = DataCollector()
        dp1 = DataPoint('A', 'https://a.com', 'T1', 'S1', '2023', 0.8, 'qualitative')
        dp2 = DataPoint('A', 'https://a.com', 'T1', 'S1', '2023', 0.8, 'qualitative')
        dp3 = DataPoint('B', 'https://b.com', 'T2', 'S2', '2023', 0.7, 'qualitative')
        result = dc._deduplicate([dp1, dp2, dp3])
        assert len(result) == 2

    def test_rate_limiter_init(self):
        from agent.modules.data_collector import DataCollector, SOURCE_RATE_LIMITS
        dc = DataCollector()
        for source in SOURCE_RATE_LIMITS:
            assert source in dc._rate_limiters

    def test_retry_handler_init(self):
        from agent.modules.data_collector import DataCollector, SOURCE_RETRY_CONFIG
        dc = DataCollector()
        for source in SOURCE_RETRY_CONFIG:
            assert source in dc._retry_handlers

    @pytest.mark.asyncio
    async def test_collect_returns_list(self):
        from agent.modules.data_collector import DataCollector
        dc = DataCollector(timeout=5)
        with patch.object(dc, '_collect_world_bank', return_value=[]), \
             patch.object(dc, '_collect_imf', return_value=[]), \
             patch.object(dc, '_collect_arxiv', return_value=[]), \
             patch.object(dc, '_collect_semantic_scholar', return_value=[]), \
             patch.object(dc, '_collect_duckduckgo', return_value=[]):
            points = await dc.collect('test query', sector='tech')
            assert isinstance(points, list)


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_burst(self):
        from agent.modules.data_collector import RateLimiter
        rl = RateLimiter(rps=100.0, burst=10)
        for _ in range(10):
            await rl.acquire()


class TestRetryHandler:
    @pytest.mark.asyncio
    async def test_success(self):
        from agent.modules.data_collector import RetryHandler
        rh = RetryHandler(max_retries=3, backoff_base=0.01, backoff_max=0.1)
        async def succeed():
            return 'ok'
        result = await rh.execute(succeed)
        assert result == 'ok'

    @pytest.mark.asyncio
    async def test_exhausted(self):
        from agent.modules.data_collector import RetryHandler
        rh = RetryHandler(max_retries=2, backoff_base=0.01, backoff_max=0.1)
        async def always_fail():
            raise ConnectionError('persistent')
        with pytest.raises(RuntimeError, match='retries exhausted'):
            await rh.execute(always_fail)


class TestFrameworkAnalyzer:
    def test_heuristic_porter(self):
        from agent.modules.framework_analyzer import FrameworkAnalyzer
        assert 'PORTER' in FrameworkAnalyzer()._heuristic_select('competitive landscape rivals', 'tech')

    def test_heuristic_pestel(self):
        from agent.modules.framework_analyzer import FrameworkAnalyzer
        assert 'PESTEL' in FrameworkAnalyzer()._heuristic_select('regulatory policy macro', 'fintech')

    def test_heuristic_swot(self):
        from agent.modules.framework_analyzer import FrameworkAnalyzer
        assert 'SWOT' in FrameworkAnalyzer()._heuristic_select('general strategy analysis', 'tech')

    def test_fallback_swot(self):
        from agent.modules.framework_analyzer import FrameworkAnalyzer
        r = FrameworkAnalyzer()._fallback_framework('SWOT', 'test')
        assert 'strengths' in r.data

    def test_fallback_porter(self):
        from agent.modules.framework_analyzer import FrameworkAnalyzer
        r = FrameworkAnalyzer()._fallback_framework('PORTER', 'test')
        assert 'competitive_rivalry' in r.data

    def test_fallback_pestel(self):
        from agent.modules.framework_analyzer import FrameworkAnalyzer
        r = FrameworkAnalyzer()._fallback_framework('PESTEL', 'test')
        assert 'political' in r.data

    def test_extract_json(self):
        from agent.modules.framework_analyzer import FrameworkAnalyzer
        r = FrameworkAnalyzer()._extract_json('```json\n{"a": 1}\n```')
        assert 'a' in r


class TestTrendForecaster:
    def test_compute_cagr(self):
        from agent.modules.trend_forecaster import TrendForecaster
        cagr = TrendForecaster()._compute_cagr(100, 121, 2)
        assert abs(cagr - 0.10) < 0.001

    def test_cagr_zero(self):
        from agent.modules.trend_forecaster import TrendForecaster
        tf = TrendForecaster()
        assert tf._compute_cagr(0, 100, 5) == 0.0

    def test_linear_forecast_shape(self):
        from agent.modules.trend_forecaster import TrendForecaster
        r = TrendForecaster()._linear_forecast([2020, 2021, 2022], [100.0, 110.0, 120.0],
            'Market Size', 'billion USD')
        assert len(r.forecast_years) == 3

    def test_linear_single_point(self):
        from agent.modules.trend_forecaster import TrendForecaster
        r = TrendForecaster()._linear_forecast([2023], [100.0], 'M', 'USD')
        assert r.method == 'Linear (insufficient data)'

    @pytest.mark.asyncio
    async def test_forecast_with_data(self, sample_data_points):
        from agent.modules.trend_forecaster import TrendForecaster
        result = await TrendForecaster().forecast(sample_data_points)
        assert result is not None


class TestReportGenerator:
    def test_fallback_summary(self, tmp_path):
        from agent.modules.report_generator import ReportGenerator
        s = ReportGenerator(output_dir=str(tmp_path))._fallback_executive_summary('AI', 'tech', 'global', 'SWOT', '3% CAGR')
        assert len(s) > 50

    def test_fallback_recs(self, tmp_path):
        from agent.modules.report_generator import ReportGenerator
        r = ReportGenerator(output_dir=str(tmp_path))._fallback_recommendations('fintech')
        assert '1.' in r

    def test_quality_gates_min(self, tmp_path):
        from agent.modules.report_generator import ReportGenerator
        from agent.modules.data_collector import DataPoint
        rg = ReportGenerator(output_dir=str(tmp_path))
        single = [DataPoint('A', 'https://a.com', 'T', 'S', '2023', 0.8, 'qualitative')]
        report = '## Strategic Analysis\n\nSWOT.\n\n## Sources & Data\n\n[1] Test\n\n' * 100
        passed, total = rg._run_quality_gates(report, single, None)
        assert passed < total


class TestMemoryManager:
    def test_save_retrieve(self, memory_manager):
        from agent.memory.memory_manager import AnalysisRecord
        r = AnalysisRecord('EV market', 'ev', 'global', ['SWOT'], 12, '/o.md', 0.78, 1200, 0.045)
        aid = memory_manager.save_analysis(r)
        assert aid > 0
        assert len(memory_manager.get_recent_analyses(limit=5)) == 1

    def test_paper_dedup(self, memory_manager):
        memory_manager.mark_paper_known('Test Paper', 'arXiv', 'doi:123')
        assert memory_manager.is_known_paper('Test Paper', 'doi:123')
        assert not memory_manager.is_known_paper('Other', 'doi:456')

    def test_llm_cost(self, memory_manager):
        memory_manager.log_llm_cost('claude', 'opus-4', 'fw', 1000, 300, 0.038)
        assert memory_manager.get_cost_summary(days=30)["total_cost_usd"] > 0

    def test_stats_empty(self, memory_manager):
        assert memory_manager.get_stats()["total_analyses"] == 0


class TestLLMClient:
    def test_privacy_mode(self, monkeypatch):
        monkeypatch.setenv('PRIVACY_MODE', '1')
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test_key')
        from tools.llm_client import UnifiedLLMClient
        c = UnifiedLLMClient()
        c._privacy_mode = True
        assert c._build_provider_chain()[0][0] == 'ollama'

    def test_cost_calc(self):
        from tools.llm_client import UnifiedLLMClient
        assert UnifiedLLMClient()._calc_cost('claude-opus-4-8', 1000, 300) > 0
        assert UnifiedLLMClient()._calc_cost('unknown', 1000, 500) == 0.0

    @pytest.mark.asyncio
    async def test_complete_fallback(self, monkeypatch):
        monkeypatch.setenv('PRIVACY_MODE', '0')
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        from tools.llm_client import UnifiedLLMClient
        c = UnifiedLLMClient()
        async def fail(*a, **k):
            raise RuntimeError('Network error')
        c._call_with_retry = fail
        r = await c.complete('test')
        assert r.text == ''
        assert r.provider == 'none'


class TestHFModelManager:
    def test_tfidf_shape(self):
        from tools.hf_model_manager import HFModelManager
        assert HFModelManager()._tfidf_fallback_encode(['a', 'b']).shape == (2, 1024)

    def test_tfidf_normalized(self):
        import numpy as np
        from tools.hf_model_manager import HFModelManager
        v = HFModelManager()._tfidf_fallback_encode(['test'])
        assert abs(np.linalg.norm(v[0]) - 1.0) < 0.01

    def test_rerank_empty(self):
        from tools.hf_model_manager import HFModelManager
        assert HFModelManager().rerank('test', [], top_k=5) == []

    def test_summarize_short(self):
        from tools.hf_model_manager import HFModelManager
        assert HFModelManager().summarize('Short.', max_length=50) == 'Short.'

    def test_singleton(self):
        from tools.hf_model_manager import get_instance
        assert get_instance() is get_instance()


class TestCitationFeedClient:
    def test_fingerprint(self):
        from agent.cross_agent.citation_feed import CitationEntry
        e1 = CitationEntry(title='T', authors='A', year=2023, source_name='X',
            url='https://x.com', abstract='Abs', citation_type='academic', confidence=0.8)
        assert len(e1.fingerprint) == 64

    def test_to_dict(self):
        from agent.cross_agent.citation_feed import CitationEntry
        d = CitationEntry(title='T', authors='A', year=2024, source_name='WB',
            url='https://wb.org', abstract='Abs', citation_type='stat', confidence=0.9, sector='fin').to_dict()
        assert d['title'] == 'T'

    def test_convert_data_points(self, sample_data_points):
        from agent.cross_agent.citation_feed import CitationFeedClient
        c = CitationFeedClient(base_url='http://localhost:99999')
        citations = c.convert_data_points_to_citations(sample_data_points)
        assert len(citations) == len(sample_data_points)

    @pytest.mark.asyncio
    async def test_health_unreachable(self):
        from agent.cross_agent.citation_feed import CitationFeedClient
        assert await CitationFeedClient(base_url='http://localhost:99999').check_health() is False


class TestBenchmarkInstrument:
    def test_start_end(self):
        from agent.cross_agent.benchmark_instrument import BenchmarkInstrument
        inst = BenchmarkInstrument()
        inst.start_run('t1', 'test query')
        s = inst.end_run('t1', {'sources_count': 10, 'frameworks_applied': ['SWOT']})
        assert s['run_id'] == 't1'

    def test_track_llm(self):
        from agent.cross_agent.benchmark_instrument import BenchmarkInstrument
        inst = BenchmarkInstrument()
        inst.start_run('t2', 'q')
        inst.track_llm_call('t2', 'claude', 'opus', 'fw', 1000, 500, 1200.0, 0.05)
        s = inst.end_run('t2', {})
        assert s['total_tokens_in'] == 1000

    def test_clear(self):
        from agent.cross_agent.benchmark_instrument import BenchmarkInstrument
        inst = BenchmarkInstrument()
        inst.start_run('c1', 'q')
        inst.end_run('c1', {})
        inst.clear_history()
        assert inst.get_all_runs() == []


class TestPrometheusExporter:
    def test_empty(self):
        from agent.cross_agent.prometheus_exporter import PrometheusExporter
        assert isinstance(PrometheusExporter().generate_metrics(), str)

    def test_record_analysis(self):
        from agent.cross_agent.prometheus_exporter import PrometheusExporter
        exp = PrometheusExporter()
        exp.record_analysis_complete(sector='fin', elapsed_seconds=5.2, sources_count=15,
            frameworks=['SWOT'], confidence_avg=0.78, word_count=2500,
            quality_gates_passed=7, quality_gates_total=7, forecast_method='Prophet', cost_usd=0.05)
        out = exp.generate_metrics()
        assert 'market_analyses_total' in out
        assert 'sector="fin"' in out

    def test_record_source(self):
        from agent.cross_agent.prometheus_exporter import PrometheusExporter
        exp = PrometheusExporter()
        exp.record_source_collection('world_bank', 5, 1200.0)
        exp.record_source_collection('arxiv', 0, 3000.0)
        out = exp.generate_metrics()
        assert 'market_source_requests_total' in out
        assert 'market_source_empty_results_total' in out

    def test_with_memory(self, memory_manager):
        from agent.cross_agent.prometheus_exporter import PrometheusExporter
        out = PrometheusExporter(memory_manager=memory_manager).generate_metrics()
        assert 'market_db_total_analyses' in out


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path, sample_data_points):
        from agent.orchestrator import MarketIntelligenceOrchestrator
        config = {'output': {'dir': str(tmp_path)}, 'memory': {'db_path': str(tmp_path / 'test.db')}}
        orch = MarketIntelligenceOrchestrator(config=config)
        from agent.modules.data_collector import DataCollector
        orch._collector = DataCollector()
        orch._collector.collect = AsyncMock(return_value=sample_data_points)
        result = await orch.analyze(query='fintech Vietnam', sector='fintech', geo_scope='Vietnam')
        assert result['status'] == 'success'
        assert Path(result["report_path"]).exists()

    @pytest.mark.asyncio
    async def test_knowledge_update(self, tmp_path):
        from agent.orchestrator import MarketIntelligenceOrchestrator
        config = {'memory': {'db_path': str(tmp_path / 'test.db')}}
        orch = MarketIntelligenceOrchestrator(config=config)
        orch._knowledge_updater = MagicMock()
        orch._knowledge_updater.run_update = MagicMock(return_value=5)
        result = await orch.update_knowledge()
        assert result['papers_added'] == 5

    def test_cross_agent_status(self):
        from fastapi.testclient import TestClient
        from agent.main import app
        resp = TestClient(app).get("/api/v1/cross-agent/status")
        assert resp.status_code == 200
        assert 'academic_research_enhanced' in resp.json()


class TestCLISmoke:
    def test_health(self):
        from fastapi.testclient import TestClient
        from agent.main import app
        assert TestClient(app).get("/health").status_code == 200

    def test_metrics(self):
        from fastapi.testclient import TestClient
        from agent.main import app
        resp = TestClient(app).get("/metrics")
        assert resp.status_code == 200
        assert 'market_analyses_total' in resp.text

    def test_stats(self):
        from fastapi.testclient import TestClient
        from agent.main import app
        assert TestClient(app).get("/api/v1/stats").status_code == 200

    def test_cost(self):
        from fastapi.testclient import TestClient
        from agent.main import app
        resp = TestClient(app).get("/api/v1/cost?days=7")
        assert resp.status_code == 200
        assert 'total_cost_usd' in resp.json()

    def test_benchmark_stats(self):
        from fastapi.testclient import TestClient
        from agent.main import app
        assert TestClient(app).get("/api/v1/benchmark/stats").status_code == 200


class TestLiveAPIIntegration:
    @pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason='ANTHROPIC_API_KEY not set')
    @pytest.mark.asyncio
    async def test_live_claude(self):
        from tools.llm_client import UnifiedLLMClient
        r = await UnifiedLLMClient().complete("What is SWOT? One sentence.", max_tokens=50, task="test")
        assert r.text
        assert r.provider == 'claude'

    @pytest.mark.skipif(not os.getenv('RUN_LIVE_API_TESTS'), reason='RUN_LIVE_API_TESTS not set')
    @pytest.mark.asyncio
    async def test_live_world_bank(self):
        from agent.modules.data_collector import DataCollector
        points = await DataCollector(timeout=15)._collect_world_bank('GDP', 'default', 'global', (2020, 2024))
        assert isinstance(points, list)

    @pytest.mark.skipif(not os.getenv('RUN_LIVE_API_TESTS'), reason='RUN_LIVE_API_TESTS not set')
    @pytest.mark.asyncio
    async def test_live_arxiv(self):
        from agent.modules.data_collector import DataCollector
        points = await DataCollector(timeout=15)._collect_arxiv('ML market', 'tech')
        assert isinstance(points, list)

    @pytest.mark.skipif(not os.getenv('RUN_LIVE_API_TESTS'), reason='RUN_LIVE_API_TESTS not set')
    @pytest.mark.asyncio
    async def test_live_imf(self):
        from agent.modules.data_collector import DataCollector
        points = await DataCollector(timeout=15)._collect_imf('GDP', 'default', 'global', (2020, 2024))
        assert isinstance(points, list)