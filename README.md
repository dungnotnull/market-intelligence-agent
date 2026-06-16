<div align="center">

# 🧠 Market Intelligence Agent

**Autonomous market research & competitive intelligence — from raw data to cited professional reports**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-59%20passed-success.svg)](tests/)

[📖 Documentation](#documentation) · [🚀 Quick Start](#quick-start) · [⚙️ Configuration](#configuration) · [🧩 Architecture](#architecture) · [🔌 API Reference](#api-reference) · [🤝 Contributing](#contributing)

</div>

---

## What It Does

Give the agent **any sector or domain query**, and it autonomously:

1. **Crawls ≥3 authoritative global sources** — World Bank, IMF, arXiv, Semantic Scholar, DuckDuckGo — in parallel
2. **Ranks & deduplicates** collected data using BGE-large embeddings + BGE-reranker cross-encoder
3. **Applies business analysis frameworks** — SWOT, Porter's Five Forces, PESTEL, VRIO, BCG Matrix — with LLM-grounded evidence
4. **Triangulates findings** across sources with confidence scores (0.0–1.0) per claim
5. **Forecasts market trends** via Prophet (primary) / ARIMA (fallback) with 3-year projections and 80% confidence intervals
6. **Generates a professional Markdown report** with executive summary, framework analysis, trend forecast, strategic recommendations, and numbered citations
7. **Self-improves weekly** by crawling new research papers into its knowledge base

**One query → a cited market intelligence report in minutes instead of days.**

---

## Example

```bash
# CLI
python -m agent.main analyze "EV battery market in Southeast Asia" --sector ev --geo "Southeast Asia" --depth comprehensive

# REST API
curl -X POST http://localhost:8007/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Vietnam fintech regulation landscape", "sector": "fintech", "geo_scope": "Vietnam", "frameworks": ["PESTEL"]}'
```

**Output:** A structured Markdown report with:
- Executive summary with key market insight
- SWOT / Porter / PESTEL / VRIO / BCG analysis with evidence-backed claims
- 3-year trend forecast with CAGR and confidence intervals
- 3–5 strategic recommendations
- Numbered source citations from World Bank, IMF, arXiv, Semantic Scholar
- Confidence & limitations disclosure

---

## Quick Start

### Prerequisites

- Python 3.12+
- At least one LLM API key (Claude or OpenAI) or Ollama for offline mode
- (Optional) CUDA GPU for faster model inference

### 1. Clone & Install

```bash
git clone https://github.com/dungnotnull/market-intelligence-agent.git
cd market-intelligence-agent
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp config/.env.example .env
# Edit .env with your API keys:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# OLLAMA_BASE_URL=http://localhost:11434  # for offline/privacy mode
```

### 3. Run

```bash
# Start the server
python -m agent.main serve --host 0.0.0.0 --port 8007

# Or run a single analysis via CLI
python -m agent.main analyze "global renewable energy market size forecast" --sector energy
```

### Docker (Recommended for Production)

```bash
# Build and run
docker compose -f docker/docker-compose.yml up -d

# With Ollama for offline mode
docker compose -f docker/docker-compose.yml --profile ollama up -d

# With GPU support
docker compose -f docker/docker-compose.yml --profile gpu up -d
```

---

## Architecture

```
User Query (sector + depth + frameworks)
        │
┌──────────────────────────────────────────────────────────────────┐
│  MarketIntelligenceOrchestrator                                  │
│                                                                  │
│  Step 1: Parse query → extract sector, geo, depth, frameworks    │
│                                                                  │
│  Step 2: DataCollector.collect()                                 │
│    → World Bank API  → IMF SDMX API                              │
│    → arXiv API       → Semantic Scholar API                       │
│    → DuckDuckGo search (fallback)                                │
│    ↓ raw DataPoint[]  (SHA256 dedup, rate-limited, retried)      │
│                                                                  │
│  Step 3: HFModelManager.encode() + BGE-reranker                  │
│    → Embed snippets → FAISS IndexFlatIP → top-20                 │
│    → Cross-encoder rerank → top-8 per framework                 │
│                                                                  │
│  Step 4: FrameworkAnalyzer.select_frameworks()                   │
│    → LLM selects: SWOT / Porter / PESTEL / VRIO / BCG          │
│                                                                  │
│  Step 5: FrameworkAnalyzer.apply_frameworks()                    │
│    → Per framework: LLM JSON structured output                   │
│    → Triangulation: cross-validate ≥3 sources                   │
│    → Confidence score per claim: 0.0–1.0                         │
│                                                                  │
│  Step 6: TrendForecaster.forecast()                              │
│    → Extract time-series → Prophet 3yr projection               │
│    → ARIMA(2,1,2) fallback → Linear regression fallback          │
│                                                                  │
│  Step 7: ReportGenerator.generate()                              │
│    → BART-CNN summarize → LLM executive summary                 │
│    → Markdown assembly + numbered citations                      │
│                                                                  │
│  Step 8: Quality gates (7 checks enforced)                      │
│                                                                  │
│  Cross-Agent:                                                    │
│    → CitationFeedClient → academic-research-enhanced             │
│    → BenchmarkInstrument → ai-benchmark-agent                   │
│    → PrometheusExporter → dockprom-enhanced                      │
└──────────────────────────────────────────────────────────────────┘
        │
  Professional Market Intelligence Report
  (Markdown + JSON metadata + SQLite record)
```

---

## Data Sources

| Source | Type | API | Coverage |
|--------|------|-----|----------|
| **World Bank Open Data** | Macroeconomic indicators | `api.worldbank.org/v2/` | GDP, sector stats, 200+ countries |
| **IMF SDMX REST** | Monetary/fiscal data | `dataservices.imf.org/REST/` | CPI, trade, exchange rates, WEO |
| **arXiv** | Academic research | `export.arxiv.org/api/query` | cs.IR, cs.AI, stat.AP, cs.LG |
| **Semantic Scholar** | Academic papers + citations | `api.semanticscholar.org/graph/v1/` | 200M+ papers |
| **DuckDuckGo** | News/industry reports | `duckduckgo_search` Python lib | Real-time search fallback |

All source APIs have **per-source rate limiting** (token-bucket) and **exponential backoff retry** with jitter.

---

## Business Analysis Frameworks

| Framework | Best For | Output Format |
|-----------|----------|---------------|
| **SWOT** | General strategic positioning | 4 quadrants × 2–4 items, each with confidence + source |
| **Porter's Five Forces** | Competitive dynamics | 5 forces scored 1–5 with intensity + evidence |
| **PESTEL** | Macro-environmental / regulatory | 6 categories × 1–3 factors with impact ratings |
| **VRIO** | Sustainable competitive advantage | Value/Rarity/Imitability/Organization assessment |
| **BCG Matrix** | Portfolio / multi-segment | Stars/Cash Cows/Question Marks/Dogs positioning |

The LLM auto-selects 1–3 frameworks based on query type, or you can specify explicitly.

---

## LLM Provider Chain

```
Claude (opus-4-8) → GPT-4o → Ollama (llama3)
   Primary          Fallback   Offline/Privacy
```

- **Automatic fallback** if primary provider fails
- **3-retry exponential backoff** (1s → 2s → 4s) per provider
- **PRIVACY_MODE=1** forces all calls through Ollama (no data leaves your machine)
- **Cost tracking** per call with `COST_PER_1K` rate table

---

## HuggingFace Models

| Model | Task | Why Chosen |
|-------|------|-----------|
| `BAAI/bge-large-en-v1.5` | Document embeddings | #1 BEIR leaderboard (NDCG@10=0.541) |
| `BAAI/bge-reranker-large` | Cross-encoder reranking | +9pp NDCG@10 over bi-encoder |
| `sentence-transformers/all-MiniLM-L6-v2` | Fast source scoring | 5× faster than BGE-large |
| `facebook/bart-large-cnn` | Source summarization | Standard abstractive summarizer |

- **CUDA auto-detect** with CPU fallback
- **600s idle model unload** to conserve memory
- **TF-IDF fallback** for all operations when models unavailable

---

## Trend Forecasting Pipeline

```
DataPoints → Extract numeric time-series
                │
        ┌───────┴───────┐
        │  Prophet fit  │  (primary: handles seasonality, missing data)
        └───────┬───────┘
                │ if Prophet fails
        ┌───────┴───────┐
        │  ARIMA(2,1,2) │  (statistical fallback)
        └───────┬───────┘
                │ if ARIMA fails
        ┌───────┴───────┐
        │  Linear reg.  │  (minimum data fallback)
        └───────┬───────┘
                │ if <2 numeric points
        ┌───────┴───────┐
        │ Qualitative   │  (text sentiment-based estimate)
        └───────────────┘
```

Each forecast includes: 3-year projection, 80% CI bands, historical + projected CAGR.

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | One of these | — | Claude API key (primary LLM) |
| `OPENAI_API_KEY` | required | — | OpenAI API key (fallback LLM) |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | No | `llama3` | Ollama model name |
| `PRIVACY_MODE` | No | `0` | Set `1` to force Ollama-only |
| `HF_TOKEN` | No | — | HuggingFace Hub token |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `PORT` | No | `8007` | Server port |

### Agent Config (`config/agent_config.yaml`)

Full configuration for data collection timeouts, rate limits, retry behavior, framework selection, trend forecasting parameters, report quality gates, and knowledge updater schedule. See [`config/agent_config.yaml`](config/agent_config.yaml).

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/analyze` | Run full market intelligence analysis |
| `GET` | `/api/v1/analyses` | List recent analyses |
| `GET` | `/api/v1/analyses/{id}` | Get specific analysis |
| `POST` | `/api/v1/knowledge/update` | Trigger knowledge base update |
| `GET` | `/api/v1/stats` | Agent statistics |
| `GET` | `/api/v1/cost` | LLM cost report |

### Cross-Agent Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/citations/feed` | Feed citations to academic-research-enhanced |
| `GET` | `/api/v1/cross-agent/status` | Cross-agent integration health |

### Benchmark Endpoints (ai-benchmark-agent)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/benchmark/stats` | Aggregate benchmark statistics |
| `GET` | `/api/v1/benchmark/runs` | List benchmark runs |
| `POST` | `/api/v1/benchmark/run` | Execute benchmarked analysis |
| `GET` | `/api/v1/benchmark/runs/{run_id}` | Run details |
| `DELETE` | `/api/v1/benchmark/history` | Clear benchmark history |

### Monitoring (dockprom-enhanced)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/metrics` | Prometheus text-format metrics |

**Key Prometheus metrics:**
- `market_analyses_total` — counter by sector
- `market_source_requests_total` — counter by source
- `market_llm_calls_total` — counter by provider/model/task
- `market_avg_confidence` — gauge by sector
- `market_analysis_duration_seconds` — histogram by sector
- `market_source_latency_seconds` — histogram by source
- `market_llm_latency_seconds` — histogram by provider

---

## Quality Gates

Every report must pass these gates before delivery:

| Gate | Threshold | Enforcement |
|------|-----------|-------------|
| Minimum sources | ≥3 unique source domains | Hard block — retry collection |
| Framework completeness | All selected frameworks populated | Retry LLM with stricter prompt |
| Confidence threshold | No claim below 0.4 without disclosure | Warn; do not suppress |
| Report length | ≥1,500 words | Retry synthesis with "expand" |
| Quantitative data | ≥1 numeric market figure | Flag if missing |
| Citation coverage | Every factual claim linked to ≥1 source | Post-generation audit |
| Forecast present | CI bands when time-series available | Fallback to qualitative |

---

## Project Structure

```
market-intelligence-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py                          # CLI + FastAPI server
│   ├── orchestrator.py                   # E2E pipeline orchestrator
│   ├── cross_agent/                      # Cross-agent integration
│   │   ├── __init__.py
│   │   ├── citation_feed.py             # → academic-research-enhanced
│   │   ├── benchmark_instrument.py      # → ai-benchmark-agent
│   │   └── prometheus_exporter.py       # → dockprom-enhanced
│   ├── memory/
│   │   ├── __init__.py
│   │   └── memory_manager.py            # SQLite WAL persistence
│   └── modules/
│       ├── __init__.py
│       ├── data_collector.py             # World Bank + IMF + arXiv + S2 + DDG
│       ├── framework_analyzer.py         # SWOT/Porter/PESTEL/VRIO/BCG
│       ├── report_generator.py           # Markdown report assembly
│       └── trend_forecaster.py           # Prophet/ARIMA/Linear forecast
├── tools/
│   ├── __init__.py
│   ├── llm_client.py                    # Claude → OpenAI → Ollama chain
│   ├── hf_model_manager.py              # BGE + MiniLM + BART-CNN manager
│   └── knowledge_updater.py             # Weekly ArXiv + S2 crawler
├── config/
│   ├── agent_config.yaml                # Agent configuration
│   └── .env.example                     # Environment template
├── docker/
│   ├── Dockerfile                        # python:3.12-slim, non-root
│   └── docker-compose.yml              # Agent + Ollama + GPU profiles
├── tests/
│   ├── test_agent.py                    # 59 tests across 15 classes
│   └── test-scenarios.md               # 8 end-to-end scenarios
├── PRODUCTION-CHECKLIST.md             # Deployment verification
├── SECOND-KNOWLEDGE-BRAIN.md           # Self-improving knowledge base
├── PROJECT-DEVELOPMENT-PHASE-TRACKING.md # Phase tracking (8/8 done)
├── requirements.txt                     # Pinned dependencies
├── CLAUDE.md                            # Agent specification
├── PROJECT-detail.md                    # Detailed project docs
└── README.md                            # This file
```

---

## Test Scenarios

```bash
# Run all tests
pytest tests/test_agent.py -v

# Run specific test classes
pytest tests/test_agent.py::TestDataCollector -v
pytest tests/test_agent.py::TestFrameworkAnalyzer -v
pytest tests/test_agent.py::TestTrendForecaster -v
pytest tests/test_agent.py::TestPrometheusExporter -v
pytest tests/test_agent.py::TestCitationFeedClient -v
pytest tests/test_agent.py::TestBenchmarkInstrument -v

# Run live API tests (requires API keys)
RUN_LIVE_API_TESTS=1 ANTHROPIC_API_KEY=sk-... pytest tests/test_agent.py::TestLiveAPIIntegration -v
```

**Test coverage:** 15 classes, 59 tests covering DataCollector, RateLimiter, RetryHandler, FrameworkAnalyzer, TrendForecaster, ReportGenerator, MemoryManager, LLMClient, HFModelManager, CitationFeedClient, BenchmarkInstrument, PrometheusExporter, Integration, CLI smoke, and Live API integration.

---

## Knowledge Base Self-Improvement

The agent maintains a **self-improving knowledge base** (`SECOND-KNOWLEDGE-BRAIN.md`):

- **Weekly crawl** every Sunday 02:00 via APScheduler
- **Sources:** ArXiv (cs.IR, cs.AI, stat.AP, cs.LG, econ.GN) + Semantic Scholar
- **Scoring:** `recency_weight=0.6 × relevance_weight=0.4` — favors recent, domain-relevant papers
- **Deduplication:** SHA256 hash of (title + DOI) — no duplicate entries
- **Impact:** Retrieved papers are embedded and used to ground framework analysis with methodological support

```bash
# Trigger manually
python -m agent.main update-knowledge

# Or via API
curl -X POST http://localhost:8007/api/v1/knowledge/update
```

---

## Graceful Degradation

The agent is designed to **never crash** — it degrades gracefully:

| Failure | Fallback |
|---------|----------|
| All LLM providers down | Heuristic framework selection + template analysis + template executive summary |
| Prophet unavailable | ARIMA(2,1,2) fallback |
| ARIMA unavailable | Linear regression fallback |
| Insufficient numeric data | Qualitative sentiment-based trend estimate |
| HuggingFace models unavailable | TF-IDF fallback for encoding, heuristic reranking, extractive summarization |
| Source API failure | Continue with remaining sources; DuckDuckGo always available |
| Cross-agent unreachable | Log warning, skip citation feed, continue analysis |

---

## Privacy-First Mode

For air-gapped or privacy-sensitive deployments:

```bash
# Force all LLM calls through local Ollama
export PRIVACY_MODE=1
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3

python -m agent.main serve
```

No data leaves your machine. All LLM inference runs locally via Ollama.

---

## Docker Deployment

```bash
# Standard deployment
docker compose -f docker/docker-compose.yml up -d

# With Ollama (offline mode)
docker compose -f docker/docker-compose.yml --profile ollama up -d

# With GPU acceleration
docker compose -f docker/docker-compose.yml --profile gpu up -d

# Check health
curl http://localhost:8007/health

# Monitor with Prometheus
curl http://localhost:8007/metrics
```

The Docker image uses `python:3.12-slim`, runs as non-root `agentuser`, and exposes port `8007` with a 30-second health check interval.

---

## Cost Estimation

| Step | Tokens (est.) | Provider | Cost (est.) |
|------|---------------|----------|-------------|
| Framework selection | 400 in / 200 out | Claude opus | ~$0.01 |
| Framework application (×2) | 4,000 in / 1,400 out | Claude opus | ~$0.10 |
| Executive summary | 3,000 in / 400 out | Claude opus | ~$0.03 |
| Metric extraction | 500 in / 200 out | Claude sonnet | ~$0.01 |
| **Total per analysis** | **~12,000 in / 3,000 out** | — | **~$0.15–$0.45** |

Track costs via:
```bash
python -m agent.main cost-report --days 30
# Or API: GET /api/v1/cost?days=30
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add tests for your changes
4. Ensure all tests pass (`pytest tests/ -v`)
5. Commit with conventional messages
6. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for strategic analysts, consultants, and product managers who need market intelligence — fast.**

</div>
