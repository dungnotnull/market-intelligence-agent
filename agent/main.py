"""
Market Intelligence Agent — Entry Point
CLI + FastAPI server for autonomous market research and competitive intelligence.

Usage:
  python -m agent.main analyze "EV battery market Southeast Asia"
  python -m agent.main serve
  python -m agent.main update-knowledge
  python -m agent.main cost-report
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import click
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    config_path = Path("config/agent_config.yaml")
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Market Intelligence Agent",
    description="Autonomous market research & competitive intelligence — from raw data to cited professional reports",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from agent.orchestrator import MarketIntelligenceOrchestrator
        _orchestrator = MarketIntelligenceOrchestrator(config=_load_config())
    return _orchestrator


# ── Include benchmark router for ai-benchmark-agent instrumentation ────────

from agent.cross_agent.benchmark_instrument import router as benchmark_router
app.include_router(benchmark_router)


# ── Pydantic schemas ────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="Market research query", min_length=3, max_length=500)
    sector: str = Field("", description="Industry sector (auto-detected if empty)")
    geo_scope: str = Field("global", description="Geographic scope (e.g., 'Southeast Asia', 'US')")
    depth: str = Field("standard", description="Analysis depth: quick | standard | comprehensive")
    frameworks: list[str] = Field(default=[], description="Frameworks to apply: SWOT, Porter, PESTEL, VRIO, BCG")
    date_range_years: int = Field(5, description="Years of historical data to look back", ge=1, le=20)


class AnalyzeResponse(BaseModel):
    status: str
    query: str
    sector: str
    geo_scope: str
    frameworks_applied: list[str]
    sources_count: int
    confidence_avg: float
    word_count: int
    quality_gates: str
    report_path: str
    elapsed_seconds: float
    report_preview: str


class KnowledgeUpdateResponse(BaseModel):
    papers_added: int
    updated_at: str


class CostResponse(BaseModel):
    period_days: int
    total_cost_usd: float
    by_model: list[dict]


class CitationFeedRequest(BaseModel):
    source_urls: list[str] = Field(default=[], description="Specific source URLs to feed")
    sector_filter: str = Field("", description="Only feed citations matching this sector")


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "market-intelligence-agent", "version": "1.0.0"}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    orch = get_orchestrator()
    from datetime import datetime
    current_year = datetime.utcnow().year
    date_range = (current_year - request.date_range_years, current_year)
    try:
        result = await orch.analyze(
            query=request.query,
            sector=request.sector,
            geo_scope=request.geo_scope,
            depth=request.depth,
            frameworks=request.frameworks if request.frameworks else None,
            date_range=date_range,
        )
        return AnalyzeResponse(**{k: result[k] for k in AnalyzeResponse.model_fields if k in result})
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analyses")
async def list_analyses(limit: int = 10):
    orch = get_orchestrator()
    return {"analyses": orch._get_memory().get_recent_analyses(limit=limit)}


@app.get("/api/v1/analyses/{analysis_id}")
async def get_analysis(analysis_id: int):
    orch = get_orchestrator()
    recent = orch._get_memory().get_recent_analyses(limit=1000)
    for a in recent:
        if a.get("id") == analysis_id:
            return a
    raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")


@app.post("/api/v1/knowledge/update", response_model=KnowledgeUpdateResponse)
async def update_knowledge(background_tasks: BackgroundTasks):
    orch = get_orchestrator()
    result = await orch.update_knowledge()
    return KnowledgeUpdateResponse(**result)


@app.get("/api/v1/cost", response_model=CostResponse)
async def cost_report(days: int = 30):
    orch = get_orchestrator()
    return CostResponse(**orch.get_cost_report(days=days))


@app.get("/api/v1/stats")
async def stats():
    orch = get_orchestrator()
    return orch.get_stats()


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    orch = get_orchestrator()
    return orch.get_prometheus_metrics()


# ── Cross-agent: Citation feed endpoint ─────────────────────────────────────

@app.post("/api/v1/citations/feed")
async def feed_citations(request: CitationFeedRequest):
    """Feed citations from recent analyses to academic-research-enhanced agent."""
    orch = get_orchestrator()
    citation_client = orch.get_citation_feed()
    memory = orch._get_memory()
    recent = memory.get_recent_analyses(limit=50)
    if request.sector_filter:
        recent = [a for a in recent if request.sector_filter.lower() in (a.get("sector") or "").lower()]

    from agent.cross_agent.citation_feed import CitationEntry
    citations = []
    for analysis in recent[:5]:
        sources = memory.save_data_sources.__self__ if hasattr(memory.save_data_sources, '__self__') else None
        citations.append(CitationEntry(
            title=analysis.get("query", ""),
            authors="market-intelligence-agent",
            year=int(analysis.get("created_at", "")[:4] or "2024"),
            source_name="Market Intelligence Analysis",
            url=f"file://{analysis.get('report_path', '')}",
            abstract=f"Analysis of {analysis.get('query', '')} in {analysis.get('sector', '')} sector",
            citation_type="qualitative",
            confidence=analysis.get("confidence_avg", 0.5),
            sector=analysis.get("sector", ""),
            tags=["market-intelligence", analysis.get("sector", "")],
        ))

    result = await citation_client.submit_citations(citations)
    return {"status": "submitted", "details": result}


@app.get("/api/v1/cross-agent/status")
async def cross_agent_status():
    """Check cross-agent integration status."""
    orch = get_orchestrator()
    citation_client = orch.get_citation_feed()
    academic_health = await citation_client.check_health()
    return {
        "academic_research_enhanced": {
            "url": citation_client._base_url,
            "reachable": academic_health,
        },
        "ai_benchmark_agent": {
            "endpoints": ["/api/v1/benchmark/stats", "/api/v1/benchmark/runs", "/api/v1/benchmark/run"],
            "available": True,
        },
        "dockprom_enhanced": {
            "metrics_endpoint": "/metrics",
            "format": "prometheus_text",
            "available": True,
        },
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Market Intelligence Agent CLI"""


@cli.command()
@click.argument("query")
@click.option("--sector", default="", help="Industry sector")
@click.option("--geo", default="global", help="Geographic scope")
@click.option("--depth", default="standard", type=click.Choice(["quick", "standard", "comprehensive"]))
@click.option("--frameworks", multiple=True, help="Frameworks: SWOT, Porter, PESTEL, VRIO, BCG")
@click.option("--output", default="output", help="Output directory")
def analyze(query, sector, geo, depth, frameworks, output):
    """Run full market intelligence analysis."""
    from agent.orchestrator import MarketIntelligenceOrchestrator
    config = _load_config()
    if output:
        config.setdefault("output", {})["dir"] = output

    orch = MarketIntelligenceOrchestrator(config=config)

    async def run():
        return await orch.analyze(
            query=query,
            sector=sector,
            geo_scope=geo,
            depth=depth,
            frameworks=list(frameworks) if frameworks else None,
        )

    result = asyncio.run(run())
    click.echo(json.dumps({k: v for k, v in result.items() if k != "report_preview"}, indent=2))
    click.echo(f"\n--- REPORT PREVIEW ---\n{result.get('report_preview', '')}")


@cli.command("update-knowledge")
def update_knowledge_cmd():
    """Run research paper crawler and update SECOND-KNOWLEDGE-BRAIN.md."""
    from tools.knowledge_updater import KnowledgeUpdater
    updater = KnowledgeUpdater()
    count = updater.run_update()
    click.echo(f"Knowledge updated: {count} new papers added.")


@cli.command("cost-report")
@click.option("--days", default=30, help="Period in days")
def cost_report_cmd(days):
    """Print LLM cost summary."""
    from agent.memory.memory_manager import MemoryManager
    mem = MemoryManager()
    summary = mem.get_cost_summary(days=days)
    click.echo(json.dumps(summary, indent=2))


@cli.command("serve")
@click.option("--host", default="0.0.0.0", help="Host")
@click.option("--port", default=8007, type=int, help="Port")
@click.option("--start-scheduler", is_flag=True, help="Start weekly knowledge updater")
def serve(host, port, start_scheduler):
    """Start the FastAPI server."""
    if start_scheduler:
        orch = get_orchestrator()
        orch.start_scheduled_updates()
        click.echo("Weekly knowledge updater scheduled.")
    uvicorn.run("agent.main:app", host=host, port=port, reload=False)


@cli.command("stats")
def stats_cmd():
    """Print agent stats."""
    from agent.memory.memory_manager import MemoryManager
    mem = MemoryManager()
    click.echo(json.dumps(mem.get_stats(), indent=2))


@cli.command("check-cross-agents")
def check_cross_agents():
    """Check cross-agent integration health."""
    from agent.orchestrator import MarketIntelligenceOrchestrator

    async def run():
        orch = MarketIntelligenceOrchestrator(config=_load_config())
        client = orch.get_citation_feed()
        academic = await client.check_health()
        click.echo(f"academic-research-enhanced: {'OK' if academic else 'UNREACHABLE'}")
        click.echo(f"ai-benchmark-agent: endpoints available at /api/v1/benchmark/*")
        click.echo(f"dockprom-enhanced: metrics at /metrics")

    asyncio.run(run())


if __name__ == "__main__":
    cli()
