"""
Cross-agent integration module for Market Intelligence Agent.
Provides interfaces for feeding reports to academic-research-enhanced citation database,
exposing REST API for ai-benchmark-agent instrumentation, and Prometheus metrics
for dockprom-enhanced monitoring.
"""

from agent.cross_agent.citation_feed import CitationFeedClient
from agent.cross_agent.benchmark_instrument import BenchmarkInstrumentAPI
from agent.cross_agent.prometheus_exporter import PrometheusExporter

__all__ = [
    "CitationFeedClient",
    "BenchmarkInstrumentAPI",
    "PrometheusExporter",
]
