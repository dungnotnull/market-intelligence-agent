# Test Scenarios — Market Intelligence Agent

## Scenario 1: Golden Path — EV Battery Market Analysis

**Input:**
```bash
python -m agent.main analyze "EV battery market in Southeast Asia" \
  --sector "ev" --geo "Southeast Asia" --depth "comprehensive"
```

**Expected behavior:**
1. DataCollector fetches World Bank energy indicators for East Asia region
2. arXiv returns 5+ papers on battery technology and EV adoption
3. Semantic Scholar returns 3+ market analysis papers
4. Framework selection: LLM selects SWOT + PESTEL (macro-environmental relevance)
5. SWOT populated with 4 quadrants × 3 items each; all confidence ≥ 0.6
6. PESTEL populated with 6 categories × 2 items each
7. TrendForecaster extracts market size from snippets; Prophet generates 3-year projection
8. Report assembles: ≥1,500 words, ≥3 citations, executive summary, recommendations
9. Quality gates: 7/7 passed
10. Output file saved to `output/market_report_ev_*.md`

**Expected output:**
- `status: "success"`
- `frameworks_applied: ["SWOT", "PESTEL"]`
- `sources_count: ≥10`
- `word_count: ≥1500`
- `quality_gates: "7/7"`
- `confidence_avg: ≥0.6`

---

## Scenario 2: Porter's Five Forces — B2B SaaS CRM Market

**Input:**
```bash
python -m agent.main analyze "Top 5 competitors in B2B SaaS CRM" \
  --sector "tech" --geo "US" --frameworks Porter
```

**Expected behavior:**
1. User explicitly requests Porter framework — no auto-selection needed
2. DataCollector collects SaaS industry data from arXiv + DuckDuckGo
3. Porter applied: all 5 forces scored 1–5 with evidence
4. Competitive rivalry score reflects high intensity in CRM (score 4–5)
5. Report includes "overall_industry_attractiveness" summary
6. Recommendations reference Porter findings

**Expected output:**
- `frameworks_applied: ["PORTER"]`
- Porter section in report with 5 forces, each with score + intensity + evidence
- Word count ≥1,500
- Quality gates ≥6/7

---

## Scenario 3: All LLM Providers Down — Graceful Degradation

**Setup:** Set `ANTHROPIC_API_KEY=""`, `OPENAI_API_KEY=""`, `OLLAMA_BASE_URL="http://invalid:99999"`

**Input:**
```bash
python -m agent.main analyze "Global fintech market" --sector fintech
```

**Expected behavior:**
1. DataCollector still collects from World Bank + arXiv + Semantic Scholar (no LLM needed)
2. Framework selection: `_heuristic_select()` returns `["SWOT"]` (fintech → finance keywords)
3. `_apply_framework()` tries LLM, all fail, returns `_fallback_framework("SWOT", query)`
4. Fallback SWOT populated with template items (confidence 0.5)
5. TrendForecaster uses `_linear_forecast()` (no LLM needed)
6. Executive summary: `_fallback_executive_summary()` (template output)
7. Report still generated and saved
8. `confidence_avg: ~0.5` (lower than normal but present)
9. Quality gates: ≥5/7 (word count may be lower with template output)

**Expected output:**
- `status: "success"` (not error)
- Report file created
- `frameworks_applied: ["SWOT"]`
- No exception raised

---

## Scenario 4: PESTEL Analysis — Vietnam Fintech Regulation

**Input:**
```python
# via REST API
POST /api/v1/analyze
{
  "query": "Vietnam fintech regulation landscape",
  "sector": "fintech",
  "geo_scope": "Vietnam",
  "frameworks": ["PESTEL"]
}
```

**Expected behavior:**
1. `_map_geo_to_wb_code("Vietnam")` returns "VN"
2. World Bank collects finance sector data for Vietnam
3. DuckDuckGo returns recent regulatory news
4. PESTEL populated: Political (regulatory policy), Economic (GDP), Legal (compliance)
5. Confidence scores reflect source agreement
6. Report includes regulatory risk assessment

**Expected output:**
- `frameworks_applied: ["PESTEL"]`
- Report includes "Political" and "Legal" sections
- ≥2 sources from news/regulatory domains

---

## Scenario 5: Trend Forecasting — Market Size Projection

**Input:**
```bash
python -m agent.main analyze "Global renewable energy market size forecast"
```

**Expected behavior:**
1. World Bank energy indicators return historical values with years
2. TrendForecaster.`_extract_time_series()` builds DataFrame with ≥2 years
3. Prophet fits model and generates 3-year projection
4. `cagr_projected` computed and included in report
5. Forecast table shown with CI bands

**Expected output:**
- `forecast_method: "Prophet"` (or "ARIMA" if Prophet unavailable)
- `forecast_cagr: non-null float`
- Forecast table in report with year/value/CI_upper/CI_lower columns

---

## Scenario 6: Knowledge Crawler — Deduplication

**Input (run twice):**
```bash
python -m agent.main update-knowledge
python -m agent.main update-knowledge  # second run
```

**Expected behavior:**
1. First run: fetches papers from arXiv + Semantic Scholar, adds up to 10 to SECOND-KNOWLEDGE-BRAIN.md, records SHA256 hashes in DB
2. Second run: fetches same papers, all hashes already in `knowledge_hashes` table → 0 new papers added
3. No duplicate rows in SECOND-KNOWLEDGE-BRAIN.md

**Expected output:**
- First run: `papers_added: N > 0`
- Second run: `papers_added: 0`
- SECOND-KNOWLEDGE-BRAIN.md has exactly N additional rows after first run

---

## Scenario 7: BCG Matrix — ASEAN E-Commerce Portfolio

**Input:**
```bash
python -m agent.main analyze "ASEAN e-commerce market portfolio segments" \
  --sector "e-commerce" --geo "Southeast Asia" --frameworks BCG
```

**Expected behavior:**
1. BCG_MATRIX_PROMPT applied with collected evidence
2. Output JSON contains: stars, cash_cows, question_marks, dogs
3. Each segment has rationale and confidence score
4. Report section "BCG Matrix" with portfolio positioning
5. Recommendations reference BCG positioning

**Expected output:**
- `frameworks_applied: ["BCG"]`
- BCG section in report with 4 quadrant categories

---

## Scenario 8: REST API Full Integration Test

**Sequence:**
```bash
# 1. Health check
GET /health → {"status": "ok"}

# 2. Analyze
POST /api/v1/analyze {"query": "global AI infrastructure market", "sector": "tech", "depth": "standard"}
→ 200 OK with AnalyzeResponse

# 3. List analyses
GET /api/v1/analyses?limit=5
→ [{"query": ..., "sector": ..., ...}]

# 4. Cost report
GET /api/v1/cost?days=7
→ {"period_days": 7, "total_cost_usd": ..., "by_model": [...]}

# 5. Stats
GET /api/v1/stats
→ {"total_analyses": N, ...}

# 6. Prometheus metrics
GET /metrics
→ # HELP ... \n market_analyses_total N

# 7. Knowledge update
POST /api/v1/knowledge/update
→ {"papers_added": N, "updated_at": ...}
```

**Expected:** All 7 endpoints return 200 with valid JSON/text responses.
