"""
PrometheusExporter: generates Prometheus-format metrics for dockprom-enhanced
(folder 14) monitoring. Tracks analysis counts, latency histograms, source
collection stats, LLM cost gauges, knowledge base growth, and quality gates.
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Produces Prometheus text-format metrics for dockprom-enhanced integration.

    Metric naming follows Prometheus conventions:
    - _total suffix for counters
    - _seconds suffix for time histograms
    - no units in metric names (use HELP doc instead)
    """

    def __init__(self, memory_manager=None):
        self._memory = memory_manager
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._labels: dict[str, dict[str, str]] = {}

    def inc_counter(self, name: str, value: float = 1.0, labels: dict[str, str] = None):
        key = self._label_key(name, labels)
        self._counters[key] += value
        if labels:
            self._labels[key] = labels

    def set_gauge(self, name: str, value: float, labels: dict[str, str] = None):
        key = self._label_key(name, labels)
        self._gauges[key] = value
        if labels:
            self._labels[key] = labels

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] = None):
        key = self._label_key(name, labels)
        self._histograms[key].append(value)
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-500:]
        if labels:
            self._labels[key] = labels

    def record_analysis_complete(self, sector: str, elapsed_seconds: float,
                                  sources_count: int, frameworks: list[str],
                                  confidence_avg: float, word_count: int,
                                  quality_gates_passed: int, quality_gates_total: int,
                                  forecast_method: str, cost_usd: float):
        """Record all metrics for a completed analysis run."""
        self.inc_counter("market_analyses_total", labels={"sector": sector})
        self.observe_histogram("market_analysis_duration_seconds", elapsed_seconds,
                              labels={"sector": sector})
        self.set_gauge("market_sources_collected", sources_count, labels={"sector": sector})
        for fw in frameworks:
            self.inc_counter("market_frameworks_applied_total", labels={"framework": fw})
        self.set_gauge("market_avg_confidence", confidence_avg, labels={"sector": sector})
        self.set_gauge("market_report_word_count", word_count, labels={"sector": sector})
        self.set_gauge("market_quality_gates_passed", quality_gates_passed,
                       labels={"sector": sector})
        self.set_gauge("market_quality_gates_total", quality_gates_total,
                       labels={"sector": sector})
        if forecast_method and forecast_method != "None":
            self.inc_counter("market_forecasts_generated_total",
                            labels={"method": forecast_method})
        if cost_usd > 0:
            self.inc_counter("market_llm_cost_usd_total", cost_usd, labels={"sector": sector})

    def record_source_collection(self, source_name: str, points_count: int, elapsed_ms: float):
        """Record metrics for a single source collection call."""
        self.inc_counter("market_source_requests_total", labels={"source": source_name})
        self.observe_histogram("market_source_latency_seconds", elapsed_ms / 1000.0,
                              labels={"source": source_name})
        if points_count == 0:
            self.inc_counter("market_source_empty_results_total", labels={"source": source_name})

    def record_llm_call(self, provider: str, model: str, task: str,
                        tokens_in: int, tokens_out: int, latency_ms: float, cost_usd: float):
        """Record metrics for an LLM API call."""
        self.inc_counter("market_llm_calls_total",
                         labels={"provider": provider, "model": model, "task": task})
        self.observe_histogram("market_llm_latency_seconds", latency_ms / 1000.0,
                              labels={"provider": provider, "task": task})
        self.inc_counter("market_llm_tokens_in_total", tokens_in,
                         labels={"provider": provider, "model": model})
        self.inc_counter("market_llm_tokens_out_total", tokens_out,
                         labels={"provider": provider, "model": model})
        if cost_usd > 0:
            self.inc_counter("market_llm_cost_usd_total", cost_usd,
                            labels={"provider": provider, "model": model})

    def record_knowledge_update(self, papers_added: int, elapsed_seconds: float):
        """Record metrics for a knowledge base update run."""
        self.inc_counter("market_knowledge_updates_total")
        self.set_gauge("market_knowledge_papers_added", papers_added)
        self.observe_histogram("market_knowledge_update_duration_seconds", elapsed_seconds)

    def generate_metrics(self) -> str:
        """Generate complete Prometheus text-format metrics output."""
        lines = []

        # ── Counters ──────────────────────────────────────────────────────
        lines.append("# HELP market_analyses_total Total analyses completed")
        lines.append("# TYPE market_analyses_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_analyses_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_source_requests_total Total source API requests made")
        lines.append("# TYPE market_source_requests_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_source_requests_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_source_empty_results_total Source requests that returned zero results")
        lines.append("# TYPE market_source_empty_results_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_source_empty_results_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_frameworks_applied_total Total frameworks applied across all analyses")
        lines.append("# TYPE market_frameworks_applied_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_frameworks_applied_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_forecasts_generated_total Total forecasts generated by method")
        lines.append("# TYPE market_forecasts_generated_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_forecasts_generated_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_llm_calls_total Total LLM API calls made")
        lines.append("# TYPE market_llm_calls_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_llm_calls_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_llm_tokens_in_total Total input tokens consumed")
        lines.append("# TYPE market_llm_tokens_in_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_llm_tokens_in_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_llm_tokens_out_total Total output tokens generated")
        lines.append("# TYPE market_llm_tokens_out_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_llm_tokens_out_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_llm_cost_usd_total Total LLM cost in USD")
        lines.append("# TYPE market_llm_cost_usd_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_llm_cost_usd_total":
                lines.append(f"{self._format_metric(name, labels)} {round(value, 6)}")

        lines.append("# HELP market_knowledge_updates_total Total knowledge base update runs")
        lines.append("# TYPE market_knowledge_updates_total counter")
        for key, value in sorted(self._counters.items()):
            name, labels = self._parse_key(key)
            if name == "market_knowledge_updates_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        # ── Gauges ────────────────────────────────────────────────────────
        lines.append("# HELP market_sources_collected Current source collection count")
        lines.append("# TYPE market_sources_collected gauge")
        for key, value in sorted(self._gauges.items()):
            name, labels = self._parse_key(key)
            if name == "market_sources_collected":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_avg_confidence Average analysis confidence")
        lines.append("# TYPE market_avg_confidence gauge")
        for key, value in sorted(self._gauges.items()):
            name, labels = self._parse_key(key)
            if name == "market_avg_confidence":
                lines.append(f"{self._format_metric(name, labels)} {round(value, 4)}")

        lines.append("# HELP market_knowledge_papers Current knowledge base paper count")
        lines.append("# TYPE market_knowledge_papers gauge")
        if self._memory:
            stats = self._memory.get_stats()
            lines.append(f"market_knowledge_papers {stats.get('knowledge_papers', 0)}")
        else:
            for key, value in sorted(self._gauges.items()):
                name, labels = self._parse_key(key)
                if name == "market_knowledge_papers":
                    lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_knowledge_papers_added Papers added in last update")
        lines.append("# TYPE market_knowledge_papers_added gauge")
        for key, value in sorted(self._gauges.items()):
            name, labels = self._parse_key(key)
            if name == "market_knowledge_papers_added":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_quality_gates_passed Quality gates passed in last analysis")
        lines.append("# TYPE market_quality_gates_passed gauge")
        for key, value in sorted(self._gauges.items()):
            name, labels = self._parse_key(key)
            if name == "market_quality_gates_passed":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_quality_gates_total Total quality gates in last analysis")
        lines.append("# TYPE market_quality_gates_total gauge")
        for key, value in sorted(self._gauges.items()):
            name, labels = self._parse_key(key)
            if name == "market_quality_gates_total":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        lines.append("# HELP market_report_word_count Word count of last generated report")
        lines.append("# TYPE market_report_word_count gauge")
        for key, value in sorted(self._gauges.items()):
            name, labels = self._parse_key(key)
            if name == "market_report_word_count":
                lines.append(f"{self._format_metric(name, labels)} {value}")

        # ── Histograms (as summaries) ─────────────────────────────────────
        lines.append("# HELP market_analysis_duration_seconds Analysis pipeline duration")
        lines.append("# TYPE market_analysis_duration_seconds summary")
        for key, values in sorted(self._histograms.items()):
            name, labels = self._parse_key(key)
            if name == "market_analysis_duration_seconds":
                lines.extend(self._format_summary(name, labels, values))

        lines.append("# HELP market_source_latency_seconds Source API request latency")
        lines.append("# TYPE market_source_latency_seconds summary")
        for key, values in sorted(self._histograms.items()):
            name, labels = self._parse_key(key)
            if name == "market_source_latency_seconds":
                lines.extend(self._format_summary(name, labels, values))

        lines.append("# HELP market_llm_latency_seconds LLM API call latency")
        lines.append("# TYPE market_llm_latency_seconds summary")
        for key, values in sorted(self._histograms.items()):
            name, labels = self._parse_key(key)
            if name == "market_llm_latency_seconds":
                lines.extend(self._format_summary(name, labels, values))

        lines.append("# HELP market_knowledge_update_duration_seconds Knowledge update duration")
        lines.append("# TYPE market_knowledge_update_duration_seconds summary")
        for key, values in sorted(self._histograms.items()):
            name, labels = self._parse_key(key)
            if name == "market_knowledge_update_duration_seconds":
                lines.extend(self._format_summary(name, labels, values))

        # ── Database stats ────────────────────────────────────────────────
        if self._memory:
            stats = self._memory.get_stats()
            cost = self._memory.get_cost_summary(days=30)
            lines.append("# HELP market_db_total_analyses Total analyses in database")
            lines.append("# TYPE market_db_total_analyses gauge")
            lines.append(f"market_db_total_analyses {stats.get('total_analyses', 0)}")
            lines.append("# HELP market_db_total_sources Total source records in database")
            lines.append("# TYPE market_db_total_sources gauge")
            lines.append(f"market_db_total_sources {stats.get('total_sources_collected', 0)}")
            lines.append("# HELP market_llm_cost_usd_30d LLM cost in last 30 days")
            lines.append("# TYPE market_llm_cost_usd_30d gauge")
            lines.append(f"market_llm_cost_usd_30d {cost.get('total_cost_usd', 0)}")

        return "\n".join(lines) + "\n"

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _label_key(name: str, labels: dict[str, str] = None) -> str:
        if not labels:
            return name
        parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(parts)}}}"

    @staticmethod
    def _parse_key(key: str) -> tuple[str, dict[str, str]]:
        if "{" not in key:
            return key, {}
        name = key[:key.index("{")]
        labels_str = key[key.index("{") + 1:key.index("}")]
        labels = {}
        for part in labels_str.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k.strip()] = v.strip()
        return name, labels

    @staticmethod
    def _format_metric(name: str, labels: dict[str, str] = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    @staticmethod
    def _format_summary(name: str, labels: dict[str, str], values: list[float]) -> list[str]:
        if not values:
            return []
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        count = sum(sorted_vals)
        total = sum(sorted_vals)
        metric_prefix = PrometheusExporter._format_metric(name, labels)

        quantiles = [0.5, 0.9, 0.95, 0.99]
        lines = []
        for q in quantiles:
            idx = min(int(n * q), n - 1)
            q_labels = {**labels, "quantile": str(q)}
            q_metric = PrometheusExporter._format_metric(name, q_labels)
            lines.append(f"{q_metric} {sorted_vals[idx]}")
        lines.append(f'{metric_prefix}_count {n}')
        lines.append(f'{metric_prefix}_sum {round(total, 6)}')
        return lines


# ── Singleton ───────────────────────────────────────────────────────────────

_exporter: Optional[PrometheusExporter] = None


def get_exporter(memory_manager=None) -> PrometheusExporter:
    global _exporter
    if _exporter is None:
        _exporter = PrometheusExporter(memory_manager=memory_manager)
    return _exporter
