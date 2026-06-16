# CLAUDE.md — Market Intelligence Agent (Folder 7)

**Agent Name:** market-intelligence-agent
**Tagline:** Autonomous market research & competitive intelligence — from raw data to cited professional reports
**Cluster:** F — Domain Intelligence & Analytics Agents
**Build Phase:** Phase 0 — Architecture complete, implementation ready

---

## Problem Statement

Professionals across strategy, consulting, and product need deep market intelligence but face two blockers: (1) authoritative data is scattered across dozens of sources (World Bank, IMF, Statista, academic journals, industry reports), and (2) synthesizing that data into a structured analysis (SWOT, Porter's Five Forces, PESTEL) requires specialist knowledge and hours of manual work. This agent automates the entire pipeline — accepting a sector/query, autonomously crawling ≥3 authoritative global sources, applying the appropriate analytical framework, triangulating findings with confidence scores, and delivering a professionally formatted, fully cited market intelligence report.

---

## Architecture Summary

```
User Query (sector + depth + frameworks)
        ↓
┌─────────────────────────────────────────────────────────────┐
│  MarketIntelligenceOrchestrator (agent/orchestrator.py)     │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────┐  │
│  │ DataCollector  │→ │FrameworkAnalyzer │→ │ReportGen   │  │
│  └────────────────┘  └──────────────────┘  └────────────┘  │
│          ↕                    ↕                  ↕          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ TrendForecaster (Prophet/ARIMA market projections)   │   │
│  └──────────────────────────────────────────────────────┘   │
│          ↕                    ↕                  ↕          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Memory / Context (SQLite WAL + knowledge hashes)     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
        ↓              ↓              ↓
   LLM API       HuggingFace    External APIs
  (llm_client)  (hf_model_mgr)  (World Bank/IMF/arXiv)
        ↓
  Professional Market Intelligence Report (Markdown + JSON)
```

### Pipeline Steps
1. **Parse query** — extract sector, geographic scope, analysis depth, requested frameworks
2. **Source discovery** — LLM selects ≥3 authoritative sources from registry (World Bank, IMF, arXiv, Semantic Scholar, news)
3. **Data collection** — `DataCollector` crawls each source asynchronously; stores raw findings with URL, date, confidence
4. **Relevance ranking** — BGE-large embeddings + BGE-reranker to rank and de-duplicate collected data
5. **Framework selection** — `FrameworkAnalyzer` uses LLM to choose SWOT/Porter/PESTEL/VRIO/BCG Matrix based on query type
6. **Framework application** — LLM applies each selected framework using collected evidence as context
7. **Triangulation** — cross-validate findings across ≥3 sources; flag conflicts with confidence score
8. **Trend forecasting** — `TrendForecaster` extracts time-series metrics and projects via Prophet/ARIMA
9. **Report synthesis** — `ReportGenerator` assembles structured Markdown report with BART-CNN summaries + LLM narrative
10. **Quality gate** — verify ≥3 sources cited, framework completeness, no unsupported claims

---

## Module List (`agent/modules/`)

| File | Responsibility |
|------|----------------|
| `data_collector.py` | Crawl World Bank, IMF, arXiv, Semantic Scholar, news RSS; score by recency × relevance; deduplicate |
| `framework_analyzer.py` | Apply SWOT/Porter/PESTEL/VRIO/BCG Matrix via LLM; triangulate across sources; assign confidence scores |
| `report_generator.py` | Assemble structured Markdown report: executive summary, market size, frameworks, recommendations, citations |
| `trend_forecaster.py` | Extract time-series from collected data; Prophet primary, ARIMA fallback; project market size/CAGR |

---

## Tools Used (`agent/tools/` and `tools/`)

| File | Responsibility |
|------|----------------|
| `tools/knowledge_updater.py` | Weekly ArXiv cs.IR+cs.AI+stat.AP + Semantic Scholar crawl → SECOND-KNOWLEDGE-BRAIN.md |
| `tools/llm_client.py` | Unified Claude/OpenAI/Ollama client; streaming; exponential backoff; cost tracking |
| `tools/hf_model_manager.py` | Lazy-load BGE-large, MiniLM, BGE-reranker, BART-CNN; CUDA auto-detect; 600s idle unload |

---

## HuggingFace Models

| Model ID | Task | Why chosen |
|----------|------|------------|
| `BAAI/bge-large-en-v1.5` | Document embeddings for semantic search across market data | #1 BEIR leaderboard (NDCG@10=0.541); 1024-dim dense retrieval |
| `sentence-transformers/all-MiniLM-L6-v2` | Source relevance scoring, template retrieval | 384-dim; 14K samples/sec; ideal for high-volume source ranking |
| `BAAI/bge-reranker-large` | Cross-encoder reranking of top-20 retrieved snippets | +9pp NDCG@10 over bi-encoder alone; Papers with Code reranking SOTA |
| `facebook/bart-large-cnn` | Summarize long market reports and research abstracts | CNN/DailyMail fine-tuned; standard for extractive+abstractive news summarization |

---

## LLM API Integration

| Provider | Priority | Use Cases |
|----------|----------|-----------|
| Claude (`claude-opus-4-8`) | Primary | Framework application (SWOT/Porter/PESTEL), source triangulation, executive summary, recommendations |
| OpenAI (`gpt-4o`) | Fallback | Structured JSON framework outputs when Claude unavailable |
| Ollama (`llama3`) | Offline | Privacy-sensitive queries; high-volume batch analysis |

**Prompt templates (in `framework_analyzer.py`):**
- `FRAMEWORK_SELECTION_PROMPT` — selects appropriate frameworks for query type
- `SWOT_ANALYSIS_PROMPT` — populates SWOT quadrants from evidence
- `PORTER_FIVE_FORCES_PROMPT` — evaluates 5 competitive forces
- `PESTEL_PROMPT` — maps macro-environmental factors
- `SYNTHESIS_REPORT_PROMPT` — compiles full executive report

---

## Knowledge Crawl Sources

| Source | Type | Frequency |
|--------|------|-----------|
| ArXiv cs.IR, cs.AI, stat.AP | Academic papers (market analysis, IR methods) | Weekly Sunday 02:00 |
| Semantic Scholar API | Cross-domain research papers | Weekly |
| World Bank Open Data API | GDP, trade, sector indicators | Weekly |
| Papers with Code | ML/AI leaderboards for analysis tools | Weekly |

---

## Supporting Tools in `tools/`

- **`knowledge_updater.py`**: Crawls ArXiv + Semantic Scholar. Scores papers by recency×relevance, SHA256 deduplication, appends to SECOND-KNOWLEDGE-BRAIN.md. APScheduler weekly cron Sunday 02:00.
- **`llm_client.py`**: Claude/OpenAI/Ollama chain. COST_PER_1K table for 7 models. Exponential backoff 1s/2s/4s. PRIVACY_MODE env forces Ollama. Streaming support.
- **`hf_model_manager.py`**: Singleton. Lazy loads BGE-large/MiniLM/BGE-reranker/BART-CNN. CUDA auto-detect. 600s idle unload via threading.Timer. TF-IDF fallback.

---

## Active Development Tasks

- [x] Architecture design and module planning
- [x] CLAUDE.md, PROJECT-detail.md, PHASE-TRACKING.md, SECOND-KNOWLEDGE-BRAIN.md
- [x] agent/main.py (CLI + FastAPI server)
- [x] agent/orchestrator.py (E2E pipeline)
- [x] agent/modules/data_collector.py
- [x] agent/modules/framework_analyzer.py
- [x] agent/modules/report_generator.py
- [x] agent/modules/trend_forecaster.py
- [x] agent/memory/memory_manager.py
- [x] tools/knowledge_updater.py
- [x] tools/llm_client.py
- [x] tools/hf_model_manager.py
- [x] config/agent_config.yaml, .env.example
- [x] docker/docker-compose.yml
- [x] tests/test-scenarios.md (5+ scenarios)
- [x] tests/test_agent.py
- [ ] Phase 1: Data source API key acquisition and live testing
- [ ] Phase 2: Prophet model calibration on real market data
- [ ] Phase 3: Report quality evaluation (human review rubric)
