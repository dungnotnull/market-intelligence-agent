# PROJECT-detail.md — Market Intelligence Agent

## Executive Summary

The Market Intelligence Agent is a production-grade autonomous research system that accepts any sector or domain query, crawls ≥3 authoritative global sources (World Bank, IMF, arXiv, Semantic Scholar, news), applies validated business analysis frameworks (SWOT, Porter's Five Forces, PESTEL, VRIO, BCG Matrix), triangulates findings with confidence scoring, forecasts market trends via Prophet/ARIMA, and delivers a professional cited market intelligence report. The agent continuously self-improves through a weekly research paper crawler that feeds its domain knowledge base.

**Problem Statement:** Strategic analysts, consultants, and product managers spend 15–40 hours per market intelligence report manually collecting data from scattered authoritative sources, applying frameworks inconsistently, and assembling citations. This agent compresses that pipeline to minutes with quantifiable confidence scores and reproducible methodology.

---

## Target Users & Use Cases

| User | Trigger | Agent Output |
|------|---------|--------------|
| Strategy consultant | "Analyze the global EV battery market in Southeast Asia" | SWOT + Porter + PESTEL report with World Bank/IEA citations, CAGR forecast |
| Product manager | "Who are the top 5 competitors in B2B SaaS CRM?" | Competitive landscape with BCG Matrix, feature comparison, market share estimates |
| Investment analyst | "Market size and growth drivers for generative AI infrastructure" | TAM/SAM/SOM estimates with Prophet forecast, academic source citations |
| Policy researcher | "PESTEL analysis of Vietnam fintech regulation landscape" | Regulatory risk assessment with IMF, World Bank, local news sources |
| Startup founder | "Porter's Five Forces for the food delivery market in ASEAN" | Competitive intensity scores per force with evidence and strategic recommendations |

---

## Agent Architecture (ASCII Diagram)

```
User Input (sector / query / depth / frameworks)
        ↓
┌────────────────────────────────────────────────────────────────┐
│  MarketIntelligenceOrchestrator                                │
│                                                                │
│  Step 1: _parse_query()                                        │
│    → Extract sector, geo, depth, frameworks, date_range        │
│                                                                │
│  Step 2: DataCollector.collect(query)                          │
│    → World Bank API  → IMF SDMX API                           │
│    → arXiv cs.IR+cs.AI → Semantic Scholar graph API           │
│    → NewsAPI / DuckDuckGo RSS                                  │
│    ↓ raw DataPoint[]                                           │
│                                                                │
│  Step 3: HFModelManager.encode() + BGE-reranker                │
│    → Embed all snippets → FAISS IndexFlatIP                    │
│    → Retrieve top-20 per sub-query → rerank to top-8           │
│                                                                │
│  Step 4: FrameworkAnalyzer.select_frameworks(query)            │
│    → LLM selects: SWOT / Porter / PESTEL / VRIO / BCG         │
│                                                                │
│  Step 5: FrameworkAnalyzer.apply_frameworks(evidence)          │
│    → Per framework: LLM JSON structured output                 │
│    → Triangulation: cross-validate across ≥3 sources          │
│    → Confidence score per claim: 0.0–1.0                       │
│                                                                │
│  Step 6: TrendForecaster.forecast(metric_series)               │
│    → Extract time-series from data                             │
│    → Prophet yearly+weekly seasonality                         │
│    → ARIMA(2,1,2) fallback                                     │
│    → 3-year projection with 80% confidence interval           │
│                                                                │
│  Step 7: ReportGenerator.generate(frameworks, forecast)        │
│    → BART-CNN: summarize top sources                           │
│    → LLM: executive summary + recommendations                  │
│    → Assemble Markdown with inline citations                   │
│                                                                │
│  Step 8: Quality gates                                         │
│    → ≥3 unique sources cited                                   │
│    → All framework sections populated                          │
│    → No unsupported claims (confidence ≥ 0.6)                 │
│    → Report length ≥ 1,500 words                               │
│    → Forecast CI present                                       │
└────────────────────────────────────────────────────────────────┘
        ↓
  Professional Market Intelligence Report
  (Markdown file + JSON metadata + SQLite record)
```

---

## Full Module Catalog

### `agent/modules/data_collector.py` — DataCollector

**Responsibility:** Asynchronously crawl authoritative global data sources for market information. Score, deduplicate, and return structured `DataPoint` objects.

**Inputs:** `sector: str`, `query: str`, `date_range: tuple`, `geo_scope: str`

**Outputs:** `list[DataPoint]` — each with: `source_name`, `url`, `title`, `snippet`, `date`, `confidence`, `data_type` (statistical/qualitative/regulatory)

**Sources crawled:**
- World Bank Open Data API (`api.worldbank.org/v2/`) — GDP, sector growth, trade stats
- IMF SDMX REST API (`datahelp.imf.org`) — macroeconomic indicators
- arXiv API (cs.IR, cs.AI, stat.AP) — research papers on market analysis methods
- Semantic Scholar Graph API — academic papers with citation counts
- DuckDuckGo HTML scrape (via `duckduckgo_search`) — recent news, industry reports
- Free fallback data from World Bank JSON endpoints

**Quality gate:** At least 3 distinct source domains in returned results.

---

### `agent/modules/framework_analyzer.py` — FrameworkAnalyzer

**Responsibility:** Select appropriate business analysis frameworks for the query type, apply them using collected evidence, cross-validate across sources, and assign confidence scores.

**Inputs:** `query: str`, `evidence: list[DataPoint]`, `frameworks: list[str] | None`

**Outputs:** `FrameworkReport` — dict mapping framework name → structured JSON analysis

**Frameworks supported:**
- **SWOT** — Strengths, Weaknesses, Opportunities, Threats (4 quadrants × 3 evidence items each)
- **Porter's Five Forces** — 5 forces, each scored 1–5 (weak–strong) with evidence
- **PESTEL** — Political, Economic, Social, Technological, Environmental, Legal × 2–4 findings each
- **VRIO** — Value, Rarity, Imitability, Organization (for competitive advantage analysis)
- **BCG Matrix** — Stars/Cash Cows/Question Marks/Dogs positioning for product/market portfolio

**Triangulation:** For each claim, count source agreements (≥3 = high confidence ≥0.8, 2 = medium 0.6, 1 = low 0.4). Flag contradictions for explicit disclosure.

**LLM prompts:** `FRAMEWORK_SELECTION_PROMPT` (selects frameworks), `SWOT_ANALYSIS_PROMPT`, `PORTER_FIVE_FORCES_PROMPT`, `PESTEL_PROMPT`, `VRIO_PROMPT`, `BCG_MATRIX_PROMPT`

---

### `agent/modules/report_generator.py` — ReportGenerator

**Responsibility:** Assemble all analysis outputs into a professional Markdown market intelligence report with inline citations and executive summary.

**Inputs:** `FrameworkReport`, `ForecastResult`, `list[DataPoint]`, `query: str`

**Outputs:** `str` (Markdown report), `dict` (JSON metadata with sources, confidence scores, timestamp)

**Report sections:**
1. Executive Summary (LLM-synthesized, 200–300 words)
2. Market Overview (size, growth rate, key players)
3. Framework Analysis (one section per applied framework)
4. Trend Forecast (Prophet chart description + CAGR)
5. Strategic Recommendations (3–5 actionable items)
6. Source Citations (numbered bibliography)
7. Confidence & Limitations disclosure

**Quality gate:** ≥1,500 words, ≥3 citations, ≥1 quantitative data point.

---

### `agent/modules/trend_forecaster.py` — TrendForecaster

**Responsibility:** Extract numeric time-series from collected data points (market size, GDP, growth rates), apply Prophet/ARIMA forecasting, and return projected values with confidence intervals.

**Inputs:** `list[DataPoint]`

**Outputs:** `ForecastResult` — `{metric: str, historical: list, forecast: list, cagr: float, ci_80_upper: list, ci_80_lower: list, method: str}`

**Pipeline:**
1. Parse numeric values from DataPoint snippets (regex + LLM extraction)
2. Build `pd.DataFrame` with `ds` (date) and `y` (value) columns
3. `prophet.Prophet(yearly_seasonality=True)` — fit + predict 3 years ahead
4. Fallback: `statsmodels.ARIMA(order=(2,1,2))` if Prophet fails
5. Compute CAGR from first to last historical + 3-year projection

---

## HuggingFace Model Selection

| Model | Task | MTEB Score | Reason chosen |
|-------|------|-----------|---------------|
| `BAAI/bge-large-en-v1.5` | Document embeddings | BEIR NDCG@10=0.541 | Top-1 English retrieval BEIR leaderboard |
| `sentence-transformers/all-MiniLM-L6-v2` | Source relevance scoring | MTEB 56.3 | 5×faster than BGE-large for bulk scoring |
| `BAAI/bge-reranker-large` | Cross-encoder reranking | +9pp NDCG@10 | Best open reranker on BEIR |
| `facebook/bart-large-cnn` | Long-source summarization | ROUGE-2=21.3 | CNN/DM fine-tuned, standard summarization baseline |

---

## LLM API Integration Spec

| Step | Provider | Prompt | Tokens (est.) |
|------|----------|--------|---------------|
| Framework selection | Claude opus-4-8 | `FRAMEWORK_SELECTION_PROMPT` | 400 in / 200 out |
| SWOT application | Claude opus-4-8 | `SWOT_ANALYSIS_PROMPT` | 2,000 in / 800 out |
| Porter's Five Forces | Claude opus-4-8 | `PORTER_FIVE_FORCES_PROMPT` | 2,000 in / 600 out |
| PESTEL analysis | Claude opus-4-8 | `PESTEL_PROMPT` | 2,000 in / 700 out |
| Executive summary | Claude opus-4-8 | `SYNTHESIS_REPORT_PROMPT` | 3,000 in / 400 out |
| Metric extraction | Claude sonnet-4-6 | `METRIC_EXTRACTION_PROMPT` | 500 in / 200 out |

**Total token budget per analysis:** ~12,000 input + 3,000 output ≈ $0.15–$0.45 per report

**Fallback chain:** Claude opus-4-8 → gpt-4o → claude-sonnet-4-6 → llama3 (Ollama)

---

## E2E Execution Flow (Numbered Steps)

1. **CLI/API receives query:** `sector="EV battery"`, `geo="Southeast Asia"`, `depth="comprehensive"`
2. **Orchestrator parses query:** extracts keywords, geographic scope, temporal range (default: past 5 years), requested frameworks (or auto-selects)
3. **DataCollector.collect():** async gather — World Bank API (GDP, energy sector), arXiv (battery technology papers), DuckDuckGo (recent news), Semantic Scholar (market papers). Returns `list[DataPoint]` with ≥15 items.
4. **Quality check:** if `len(unique_sources) < 3`, add DuckDuckGo fallback search.
5. **Embedding + reranking:** BGE-large embeds all snippets → FAISS IndexFlatIP → top-20 by cosine → BGE-reranker cross-encoder → top-8 per framework
6. **LLM framework selection:** `FRAMEWORK_SELECTION_PROMPT` → `["SWOT", "PESTEL", "Porter"]`
7. **Parallel framework application:** asyncio.gather(SWOT_task, PESTEL_task, Porter_task) each with 4-retry LLM loop
8. **Triangulation:** for each claim in each framework, count source agreement; assign confidence 0.0–1.0; flag low-confidence (<0.6) items
9. **Trend extraction:** parse numeric market size/CAGR from DataPoints → Prophet 3-year forecast
10. **Report assembly:** BART-CNN summary of top 5 sources → LLM executive summary → Markdown assembly with numbered citations
11. **Quality gate validation:** assert ≥3 sources, ≥1 quantitative metric, ≥1,500 words, all framework sections populated
12. **Persist & return:** SQLite `analyses` table, write `report.md` to output dir, return JSON metadata

**Error handling:**
- Source API failure → log warning, continue with remaining sources; fail only if <2 sources available
- LLM failure → retry 3× with exponential backoff; fall back to next provider; use heuristic template as last resort
- Prophet failure → ARIMA(2,1,2); if both fail, report "Trend data insufficient" with raw metrics table

---

## SECOND-KNOWLEDGE-BRAIN.md Integration

- **Sources:** ArXiv cs.IR, cs.AI, stat.AP; Semantic Scholar (market research methods); Papers with Code
- **Crawl config:** Weekly Sunday 02:00 via APScheduler
- **Dedup:** SHA256 of (title+DOI); skip if already present in `knowledge_hashes` SQLite table
- **Impact:** Retrieved papers are embedded (BGE-large) and stored in `MemoryManager.papers` table. `FrameworkAnalyzer` retrieves relevant papers during framework application to cite as methodological support.

---

## Quality Gates

| Gate | Threshold | Enforcement |
|------|-----------|-------------|
| Minimum sources | ≥3 unique source domains | Hard block — retry collection if not met |
| Framework completeness | All selected frameworks fully populated | Retry LLM with stricter prompt |
| Confidence threshold | No claim below 0.4 without explicit disclosure | Warn user; do not suppress report |
| Report length | ≥1,500 words | Retry synthesis with "expand" instruction |
| Quantitative data | ≥1 numeric market size or growth figure | Flag if missing; recommend supplementary search |
| Citation coverage | Every factual claim linked to ≥1 source | Post-generation citation audit |
| Forecast present | CI bands provided when time-series data available | Fallback to qualitative trend if data missing |

---

## Test Scenarios

See `tests/test-scenarios.md` for 8 full end-to-end scenarios.

---

## Key Design Decisions

1. **Async source crawling:** `asyncio.gather` for parallel source fetching — reduces collection time from ~30s to ~8s for 4 sources.
2. **BGE-large + BGE-reranker two-stage pipeline:** Bi-encoder for recall (top-20), cross-encoder for precision (top-8). Avoids cross-encoder latency on full corpus.
3. **Prophet over ARIMA as primary:** Prophet handles missing data and seasonal patterns in macro-economic time-series better than ARIMA; ARIMA kept as fallback for robustness.
4. **Confidence scoring over binary filtering:** Low-confidence findings are disclosed rather than dropped — analysts need to know the limits of the evidence.
5. **Multi-framework support:** Not every query fits SWOT. LLM selects the most appropriate framework(s) for the query type (BCG for portfolio, VRIO for competitive moat, PESTEL for macro/regulatory).
6. **BART-CNN for source summarization:** Long source documents (PDF reports, research papers) are summarized before being passed to LLM context, staying well within context window limits.
7. **Template fallback for all LLM steps:** Every LLM prompt has a heuristic/template fallback so the agent produces a usable (if lower quality) report even with no API access.
