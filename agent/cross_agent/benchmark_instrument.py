"""
BenchmarkInstrumentAPI: exposes REST API endpoints for ai-benchmark-agent (folder 22)
to instrument LLM calls, measure latency, track token usage, and collect
performance metrics for benchmarking.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/benchmark", tags=["benchmark"])


# ── Pydantic schemas ────────────────────────────────────────────────────────

class BenchmarkRunRequest(BaseModel):
    query: str = Field(..., description="Query to benchmark", min_length=3)
    sector: str = Field("", description="Industry sector")
    geo_scope: str = Field("global", description="Geographic scope")
    depth: str = Field("standard", description="Analysis depth: quick | standard | comprehensive")
    frameworks: list[str] = Field(default=[], description="Frameworks to apply")
    repeat: int = Field(1, description="Number of benchmark iterations", ge=1, le=10)
    track_llm_calls: bool = Field(True, description="Track individual LLM call metrics")


class BenchmarkRunResult(BaseModel):
    run_id: str
    query: str
    total_elapsed_ms: float
    avg_elapsed_ms: float
    min_elapsed_ms: float
    max_elapsed_ms: float
    iterations: int
    llm_calls_tracked: int
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    data_points_collected: int
    frameworks_applied: list[str]
    quality_gates_passed: int
    quality_gates_total: int


class LLMBenchmarkDetail(BaseModel):
    provider: str
    model: str
    task: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float


class BenchmarkInstrument:
    """Instruments and tracks performance metrics for ai-benchmark-agent consumption."""

    def __init__(self, memory_manager=None):
        self._memory = memory_manager
        self._active_runs: dict[str, dict] = {}
        self._llm_call_log: dict[str, list[dict]] = {}
        self._completed_runs: list[dict] = []

    def start_run(self, run_id: str, query: str, config: dict = None) -> str:
        """Start tracking a benchmark run."""
        self._active_runs[run_id] = {
            "run_id": run_id,
            "query": query,
            "config": config or {},
            "started_at": time.monotonic(),
            "started_at_utc": datetime.utcnow().isoformat(),
            "llm_calls": [],
            "data_points_count": 0,
            "frameworks_applied": [],
            "quality_gates": (0, 0),
        }
        self._llm_call_log[run_id] = []
        return run_id

    def end_run(self, run_id: str, result: dict) -> Optional[dict]:
        """End a benchmark run and compile final metrics."""
        run = self._active_runs.pop(run_id, None)
        if not run:
            return None
        elapsed = (time.monotonic() - run["started_at"]) * 1000
        llm_calls = self._llm_call_log.pop(run_id, [])
        total_tokens_in = sum(c.get("tokens_in", 0) for c in llm_calls)
        total_tokens_out = sum(c.get("tokens_out", 0) for c in llm_calls)
        total_cost = sum(c.get("cost_usd", 0.0) for c in llm_calls)

        summary = {
            "run_id": run_id,
            "query": run["query"],
            "total_elapsed_ms": round(elapsed, 1),
            "started_at_utc": run["started_at_utc"],
            "ended_at_utc": datetime.utcnow().isoformat(),
            "llm_calls": llm_calls,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_cost_usd": round(total_cost, 6),
            "data_points_collected": result.get("sources_count", 0),
            "frameworks_applied": result.get("frameworks_applied", []),
            "quality_gates_passed": 0,
            "quality_gates_total": 0,
        }
        self._completed_runs.append(summary)
        if len(self._completed_runs) > 100:
            self._completed_runs = self._completed_runs[-100:]
        return summary

    def track_llm_call(self, run_id: str, provider: str, model: str,
                       task: str, tokens_in: int, tokens_out: int,
                       latency_ms: float, cost_usd: float):
        """Track an individual LLM call within a benchmark run."""
        if run_id not in self._llm_call_log:
            self._llm_call_log[run_id] = []
        self._llm_call_log[run_id].append({
            "provider": provider,
            "model": model,
            "task": task,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": round(cost_usd, 6),
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_run_summary(self, run_id: str) -> Optional[dict]:
        """Get summary for a specific benchmark run."""
        for run in self._completed_runs:
            if run["run_id"] == run_id:
                return run
        return None

    def get_all_runs(self, limit: int = 20) -> list[dict]:
        """Get recent benchmark runs."""
        return self._completed_runs[-limit:]

    def get_aggregate_stats(self) -> dict:
        """Get aggregate benchmark statistics across all runs."""
        if not self._completed_runs:
            return {"total_runs": 0}
        runs = self._completed_runs
        elapsed_values = [r["total_elapsed_ms"] for r in runs]
        return {
            "total_runs": len(runs),
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 1),
            "min_elapsed_ms": round(min(elapsed_values), 1),
            "max_elapsed_ms": round(max(elapsed_values), 1),
            "total_tokens_used": sum(r.get("total_tokens_in", 0) + r.get("total_tokens_out", 0) for r in runs),
            "total_cost_usd": round(sum(r.get("total_cost_usd", 0) for r in runs), 4),
        }

    def clear_history(self):
        """Clear all benchmark history."""
        self._completed_runs.clear()
        self._active_runs.clear()
        self._llm_call_log.clear()


# ── Singleton ───────────────────────────────────────────────────────────────

_instrument: Optional[BenchmarkInstrument] = None


def get_instrument(memory_manager=None) -> BenchmarkInstrument:
    global _instrument
    if _instrument is None:
        _instrument = BenchmarkInstrument(memory_manager=memory_manager)
    return _instrument


# ── FastAPI routes for ai-benchmark-agent ──────────────────────────────────

@router.get("/stats")
async def benchmark_stats():
    """Get aggregate benchmark statistics."""
    inst = get_instrument()
    return inst.get_aggregate_stats()


@router.get("/runs")
async def benchmark_runs(limit: int = 20):
    """List recent benchmark runs."""
    inst = get_instrument()
    return {"runs": inst.get_all_runs(limit=limit)}


@router.get("/runs/{run_id}")
async def benchmark_run_detail(run_id: str):
    """Get details for a specific benchmark run."""
    inst = get_instrument()
    summary = inst.get_run_summary(run_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return summary


@router.post("/run", response_model=BenchmarkRunResult)
async def execute_benchmark_run(request: BenchmarkRunRequest):
    """Execute a benchmarked analysis run with detailed instrumentation."""
    from agent.orchestrator import MarketIntelligenceOrchestrator
    from agent.memory.memory_manager import MemoryManager
    import uuid

    run_id = f"bench-{uuid.uuid4().hex[:12]}"
    inst = get_instrument()
    memory = MemoryManager()
    config = {"memory": {"db_path": "data/market_intelligence.db"}}

    orch = MarketIntelligenceOrchestrator(config=config)
    inst.start_run(run_id, request.query, config={
        "sector": request.sector,
        "geo_scope": request.geo_scope,
        "depth": request.depth,
        "repeat": request.repeat,
    })

    elapsed_times = []
    last_result = None
    for i in range(request.repeat):
        start = time.monotonic()
        result = await orch.analyze(
            query=request.query,
            sector=request.sector,
            geo_scope=request.geo_scope,
            depth=request.depth,
            frameworks=request.frameworks if request.frameworks else None,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        elapsed_times.append(elapsed_ms)
        last_result = result

    summary = inst.end_run(run_id, last_result or {})

    if not summary:
        raise HTTPException(status_code=500, detail="Benchmark run tracking failed")

    return BenchmarkRunResult(
        run_id=run_id,
        query=request.query,
        total_elapsed_ms=round(sum(elapsed_times), 1),
        avg_elapsed_ms=round(sum(elapsed_times) / len(elapsed_times), 1),
        min_elapsed_ms=round(min(elapsed_times), 1),
        max_elapsed_ms=round(max(elapsed_times), 1),
        iterations=request.repeat,
        llm_calls_tracked=len(summary.get("llm_calls", [])),
        total_tokens_in=summary.get("total_tokens_in", 0),
        total_tokens_out=summary.get("total_tokens_out", 0),
        total_cost_usd=summary.get("total_cost_usd", 0.0),
        data_points_collected=summary.get("data_points_collected", 0),
        frameworks_applied=summary.get("frameworks_applied", []),
        quality_gates_passed=summary.get("quality_gates_passed", 0),
        quality_gates_total=summary.get("quality_gates_total", 0),
    )


@router.delete("/history")
async def clear_benchmark_history():
    """Clear all benchmark run history."""
    inst = get_instrument()
    inst.clear_history()
    return {"status": "cleared"}
