# Production Checklist — Market Intelligence Agent

> Pre-deployment verification checklist for production readiness.

## Pre-Deployment

- [x] All source code reviewed and passes linting
- [x] All automated tests pass (`pytest tests/ -v`)
- [x] Docker image builds successfully (`docker build -f docker/Dockerfile -t market-intelligence-agent .`)
- [x] Docker Compose starts without errors (`docker compose -f docker/docker-compose.yml up -d`)
- [x] Health check endpoint responds (`GET /health` → `{"status": "ok"}`)
- [x] Environment variables documented in `.env.example`
- [x] No hardcoded secrets or API keys in source code
- [x] All dependencies pinned in `requirements.txt`
- [x] Python 3.12 compatibility verified
- [x] Non-root container user (`agentuser`) configured in Dockerfile

## API Configuration

- [x] At least one LLM API key configured (ANTHROPIC_API_KEY or OPENAI_API_KEY)
- [x] Ollama URL configured for offline fallback (`OLLAMA_BASE_URL`)
- [x] PRIVACY_MODE set to `1` for air-gapped deployments
- [x] HuggingFace token configured for model downloads (`HF_TOKEN`)
- [x] Rate limits configured per source in `SOURCE_RATE_LIMITS`
- [x] Retry backoff configured per source in `SOURCE_RETRY_CONFIG`

## Service Verification

- [x] `/health` returns 200 OK
- [x] `/metrics` returns Prometheus text format
- [x] `/api/v1/analyze` accepts POST and returns `AnalyzeResponse`
- [x] `/api/v1/analyses` returns analysis history
- [x] `/api/v1/stats` returns aggregate statistics
- [x] `/api/v1/cost` returns LLM cost summary
- [x] `/api/v1/knowledge/update` triggers paper crawler
- [x] `/api/v1/cross-agent/status` returns cross-agent integration health
- [x] `/api/v1/benchmark/stats` returns benchmark statistics
- [x] CORS middleware configured for cross-origin access

## Cross-Agent Integration

- [x] `CitationFeedClient` configured for `academic-research-enhanced` agent
  - Base URL: `ACADEMIC_RESEARCH_ENHANCED_URL` env var (default: `http://localhost:8018`)
  - API Key: `ACADEMIC_RESEARCH_API_KEY` env var (optional)
  - Endpoint: `POST /api/v1/citations/batch`
- [x] `BenchmarkInstrumentAPI` exposed for `ai-benchmark-agent`
  - Endpoints: `/api/v1/benchmark/stats`, `/api/v1/benchmark/runs`, `/api/v1/benchmark/run`
  - Tracks: LLM latency, token usage, cost per call
- [x] `PrometheusExporter` integrated for `dockprom-enhanced`
  - Endpoint: `GET /metrics`
  - Metrics: counters, gauges, histograms for analyses, sources, LLM, knowledge, quality

## Data Collection

- [x] World Bank Open Data API client operational (free, no key required)
- [x] IMF SDMX REST API client operational (free, no key required)
- [x] arXiv XML API client operational (free, no key required)
- [x] Semantic Scholar Graph API client operational (free tier)
- [x] DuckDuckGo search fallback operational
- [x] Rate limiting enforced per source (token-bucket algorithm)
- [x] Retry with exponential backoff per source (jittered)
- [x] SHA256 deduplication for collected DataPoints
- [x] Minimum 3 unique source domains enforced

## Framework Analysis

- [x] SWOT analysis prompt produces structured JSON
- [x] Porter's Five Forces prompt with 1–5 intensity scoring
- [x] PESTEL prompt with 6-category mapping
- [x] VRIO analysis prompt
- [x] BCG Matrix prompt
- [x] Source triangulation algorithm (confidence scoring)
- [x] Conflict detection between framework findings
- [x] Template fallbacks for all frameworks (LLM-down mode)

## Trend Forecasting

- [x] Prophet primary forecasting (yearly seasonality)
- [x] ARIMA(2,1,2) fallback
- [x] Linear regression fallback
- [x] Qualitative sentiment-based trend (minimal data)
- [x] CAGR computation (historical + projected)
- [x] 80% confidence interval extraction
- [x] Graceful handling of insufficient data

## Report Generation

- [x] BART-CNN source summarization pipeline
- [x] LLM executive summary (200–300 words)
- [x] LLM strategic recommendations (3–5 items)
- [x] Markdown assembly with section headers
- [x] Numbered citation bibliography
- [x] Confidence score disclosure section
- [x] Quality gate validation (7 gates)
- [x] JSON metadata output alongside Markdown

## Quality Gates

| Gate | Threshold | Status |
|------|-----------|--------|
| Minimum sources | ≥3 unique source domains | ✅ Enforced |
| Framework completeness | All selected frameworks populated | ✅ Enforced |
| Confidence threshold | No claim below 0.4 without disclosure | ✅ Enforced |
| Report length | ≥1,500 words | ✅ Enforced |
| Quantitative data | ≥1 numeric market figure | ✅ Warned |
| Citation coverage | Every factual claim linked to ≥1 source | ✅ Enforced |
| Forecast present | CI bands when time-series available | ✅ Enforced |

## Monitoring & Observability

- [x] Prometheus metrics exported at `/metrics`
- [x] Counter: `market_analyses_total` by sector
- [x] Counter: `market_source_requests_total` by source
- [x] Counter: `market_llm_calls_total` by provider/model/task
- [x] Gauge: `market_avg_confidence` by sector
- [x] Gauge: `market_knowledge_papers`
- [x] Histogram: `market_analysis_duration_seconds` by sector
- [x] Histogram: `market_source_latency_seconds` by source
- [x] Histogram: `market_llm_latency_seconds` by provider

## Knowledge Base

- [x] APScheduler weekly cron (Sunday 02:00)
- [x] ArXiv crawler (cs.IR, cs.AI, stat.AP, cs.LG, econ.GN)
- [x] Semantic Scholar crawler
- [x] SHA256 deduplication for paper entries
- [x] SECOND-KNOWLEDGE-BRAIN.md append with structured format
- [x] Recency × relevance paper scoring

## Memory & Persistence

- [x] SQLite WAL mode for concurrent access
- [x] 5 tables: analyses, data_sources, frameworks, llm_cost_log, knowledge_hashes
- [x] Thread-safe access with locking
- [x] Cost tracking per LLM call

## Security

- [x] No API keys in source code (env vars only)
- [x] `.env.example` documents all required variables
- [x] Non-root Docker container user
- [x] CORS middleware configured
- [x] PRIVACY_MODE forces Ollama-only LLM calls
- [x] No network exposure beyond port 8007

## Operational

- [x] Docker health check configured (30s interval)
- [x] Graceful degradation when LLM providers unavailable
- [x] Template fallbacks for all LLM-dependent operations
- [x] TF-IDF fallback for all HuggingFace model operations
- [x] 600s idle model unload to conserve memory
- [x] CUDA auto-detect with CPU fallback

---

**Sign-off:** This checklist confirms the Market Intelligence Agent is production-ready
for deployment with all Phase 0–8 tasks completed.
