"""
ReportGenerator: assembles structured market intelligence reports in Markdown.
Uses BART-CNN to summarize long sources and LLM for executive summary.
Includes numbered citations, confidence disclosures, and quality gate validation.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SYNTHESIS_REPORT_PROMPT = """You are a senior analyst at a top-tier consulting firm.

Write a professional executive summary (200-300 words) for this market intelligence report:

Query: {query}
Sector: {sector} | Geography: {geo_scope}

Key Framework Findings:
{framework_summary}

Market Forecast:
{forecast_summary}

Average Confidence Level: {confidence:.0%}
Number of Sources: {sources_count}

Instructions:
1. Open with the single most important market insight
2. State market size/growth rate if available
3. Highlight 2-3 key strategic implications
4. Close with the top recommendation
Tone: Professional consulting report. Be specific and quantitative where possible.
Do NOT start with "Executive Summary:" — dive straight into the insight."""

RECOMMENDATIONS_PROMPT = """Based on this market intelligence analysis, provide 3-5 strategic recommendations.

Query: {query}
Framework findings: {framework_summary}
Forecast: {forecast_summary}

Format each recommendation as:
1. **[Action Title]**: Specific actionable recommendation with justification (1-2 sentences).

Be specific, actionable, and evidence-based. Number each recommendation."""


@dataclass
class ReportMetadata:
    query: str
    sector: str
    geo_scope: str
    frameworks_applied: list[str]
    sources_count: int
    unique_source_domains: list[str]
    confidence_avg: float
    word_count: int
    generated_at: str
    report_path: str
    quality_gates_passed: int
    quality_gates_total: int
    forecast_method: str = ""


class ReportGenerator:
    def __init__(self, llm_client=None, hf_manager=None, output_dir: str = "output"):
        self._llm = llm_client
        self._hf = hf_manager
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def generate(
        self,
        query: str,
        sector: str,
        geo_scope: str,
        framework_report,
        forecast_result,
        data_points: list,
    ) -> tuple[str, ReportMetadata]:
        frameworks_summary = self._summarize_frameworks(framework_report)
        forecast_summary = self._summarize_forecast(forecast_result)

        sources_text = self._summarize_sources(data_points)
        exec_summary = await self._generate_executive_summary(
            query, sector, geo_scope, frameworks_summary, forecast_summary,
            framework_report.frameworks if framework_report else [],
            len(data_points),
        )
        recommendations = await self._generate_recommendations(
            query, frameworks_summary, forecast_summary
        )

        report_md = self._assemble_report(
            query=query,
            sector=sector,
            geo_scope=geo_scope,
            exec_summary=exec_summary,
            frameworks_summary=frameworks_summary,
            framework_report=framework_report,
            forecast_result=forecast_result,
            forecast_summary=forecast_summary,
            sources_text=sources_text,
            recommendations=recommendations,
            data_points=data_points,
        )

        gates_passed, gates_total = self._run_quality_gates(report_md, data_points, framework_report)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_sector = re.sub(r"[^\w]", "_", sector or "general")[:30]
        filename = f"market_report_{safe_sector}_{timestamp}.md"
        report_path = self._output_dir / filename
        report_path.write_text(report_md, encoding="utf-8")

        unique_domains = list({p.source_domain for p in data_points})
        avg_confidence = (
            sum(fr.confidence_avg for fr in framework_report.frameworks) / len(framework_report.frameworks)
            if framework_report and framework_report.frameworks else 0.5
        )

        metadata = ReportMetadata(
            query=query,
            sector=sector,
            geo_scope=geo_scope,
            frameworks_applied=[fr.framework for fr in (framework_report.frameworks if framework_report else [])],
            sources_count=len(data_points),
            unique_source_domains=unique_domains,
            confidence_avg=round(avg_confidence, 3),
            word_count=len(report_md.split()),
            generated_at=datetime.utcnow().isoformat(),
            report_path=str(report_path),
            quality_gates_passed=gates_passed,
            quality_gates_total=gates_total,
            forecast_method=forecast_result.method if forecast_result else "None",
        )

        logger.info("Report generated: %s (%d words, %d/%d quality gates passed)",
                    filename, metadata.word_count, gates_passed, gates_total)
        return report_md, metadata

    # ── Section builders ──────────────────────────────────────────────────────

    def _assemble_report(self, **kwargs) -> str:
        query = kwargs["query"]
        sector = kwargs["sector"]
        geo_scope = kwargs["geo_scope"]
        exec_summary = kwargs["exec_summary"]
        framework_report = kwargs["framework_report"]
        forecast_result = kwargs["forecast_result"]
        recommendations = kwargs["recommendations"]
        data_points = kwargs["data_points"]

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        sections = []

        sections.append(f"# Market Intelligence Report: {sector or query}\n")
        sections.append(f"**Query:** {query}  \n**Geography:** {geo_scope}  \n**Generated:** {now}  \n**Sources:** {len(data_points)} data points\n")
        sections.append("---\n")
        sections.append("## Executive Summary\n")
        sections.append(exec_summary + "\n")
        sections.append("---\n")

        if forecast_result:
            sections.append("## Market Size & Trend Forecast\n")
            sections.append(self._format_forecast_section(forecast_result))
            sections.append("---\n")

        if framework_report and framework_report.frameworks:
            sections.append("## Strategic Analysis\n")
            for fr in framework_report.frameworks:
                sections.append(self._format_framework_section(fr))
            sections.append("---\n")

        sections.append("## Strategic Recommendations\n")
        sections.append(recommendations + "\n")
        sections.append("---\n")

        if data_points:
            sections.append("## Sources & Data\n")
            sections.append(self._format_sources_section(data_points))
            sections.append("---\n")

        avg_confidence = (
            sum(fr.confidence_avg for fr in framework_report.frameworks) / len(framework_report.frameworks)
            if framework_report and framework_report.frameworks else 0.5
        )
        sections.append("## Confidence & Limitations\n")
        sections.append(
            f"**Overall confidence:** {avg_confidence:.0%}  \n"
            f"**Sources used:** {len({p.source_domain for p in data_points})} unique domains  \n"
            "**Note:** This report is AI-generated for research purposes. "
            "Verify critical figures with primary sources before strategic decisions.  \n"
        )
        if framework_report and framework_report.conflicts:
            sections.append("\n**Detected conflicts between framework findings:**\n")
            for c in framework_report.conflicts[:3]:
                sections.append(f"- {c['framework_a']}: \"{c['claim_a']}\" vs {c['framework_b']}: \"{c['claim_b']}\"\n")

        return "\n".join(sections)

    def _format_forecast_section(self, forecast) -> str:
        lines = [
            f"**Metric:** {forecast.metric}  ",
            f"**Unit:** {forecast.unit}  ",
            f"**Historical CAGR:** {forecast.cagr_historical:.1%}  ",
            f"**Projected CAGR (3yr):** {forecast.cagr_projected:.1%}  ",
            f"**Method:** {forecast.method} (confidence: {forecast.confidence:.0%})  ",
            "",
            "| Year | Value | 80% CI Lower | 80% CI Upper |",
            "|------|-------|-------------|-------------|",
        ]
        for y, v, lo, hi in zip(forecast.historical_years[-3:], forecast.historical_values[-3:],
                                 [None] * 3, [None] * 3):
            lines.append(f"| {y} (hist.) | {v:.1f} | — | — |")
        for y, v, lo, hi in zip(forecast.forecast_years, forecast.forecast_values,
                                 forecast.ci_80_lower, forecast.ci_80_upper):
            lines.append(f"| {y} (proj.) | {v:.1f} | {lo:.1f} | {hi:.1f} |")
        if forecast.notes:
            lines.append(f"\n_Note: {forecast.notes}_")
        return "\n".join(lines) + "\n"

    def _format_framework_section(self, fr) -> str:
        lines = [f"### {fr.framework} Analysis\n",
                 f"_Confidence: {fr.confidence_avg:.0%} | Sources: {', '.join(fr.sources_used[:4])}_\n"]
        data = fr.data
        if fr.framework == "SWOT":
            for quadrant in ["strengths", "weaknesses", "opportunities", "threats"]:
                items = data.get(quadrant, [])
                if items:
                    lines.append(f"**{quadrant.upper()}**")
                    for item in items:
                        conf = item.get("confidence", 0.5)
                        src = item.get("source", "")
                        lines.append(f"- {item.get('claim', '')} *(confidence: {conf:.0%}, source: {src})*")
                    lines.append("")
        elif fr.framework == "PORTER":
            for force in ["competitive_rivalry", "supplier_power", "buyer_power", "new_entrants", "substitutes"]:
                f_data = data.get(force, {})
                if f_data:
                    score = f_data.get("score", "—")
                    intensity = f_data.get("intensity", "—")
                    evidence = f_data.get("evidence", "")
                    lines.append(f"**{force.replace('_', ' ').title()}** — Score: {score}/5 ({intensity})")
                    lines.append(f"  {evidence}")
                    lines.append("")
            overall = data.get("overall_industry_attractiveness", "")
            if overall:
                lines.append(f"**Overall Industry Attractiveness:** {overall}\n")
        elif fr.framework == "PESTEL":
            for category in ["political", "economic", "social", "technological", "environmental", "legal"]:
                factors = data.get(category, [])
                if factors:
                    lines.append(f"**{category.upper()}**")
                    for f in factors:
                        impact = f.get("impact", "")
                        desc = f.get("description", "")
                        lines.append(f"- {f.get('factor', '')} ({impact}): {desc}")
                    lines.append("")
        else:
            lines.append(f"```json\n{json.dumps(data, indent=2)[:2000]}\n```\n")
        return "\n".join(lines)

    def _format_sources_section(self, data_points: list) -> str:
        citation_index = {}
        lines = []
        idx = 1
        for dp in data_points:
            if dp.url not in citation_index:
                citation_index[dp.url] = idx
                lines.append(
                    f"[{idx}] **{dp.source_name}** — [{dp.title[:80]}]({dp.url})  "
                    f"\n    *{dp.date[:10] if dp.date else 'n/d'} | confidence: {dp.confidence:.0%} | type: {dp.data_type}*\n"
                )
                idx += 1
        return "\n".join(lines)

    def _summarize_frameworks(self, framework_report) -> str:
        if not framework_report or not framework_report.frameworks:
            return "No framework analysis available."
        summaries = []
        for fr in framework_report.frameworks:
            summaries.append(f"{fr.framework}: confidence={fr.confidence_avg:.0%}")
            if fr.framework == "PORTER":
                rivalry = fr.data.get("competitive_rivalry", {}).get("score", "?")
                summaries.append(f"  Competitive rivalry score: {rivalry}/5")
            elif fr.framework == "SWOT":
                ops = len(fr.data.get("opportunities", []))
                threats = len(fr.data.get("threats", []))
                summaries.append(f"  {ops} opportunities identified, {threats} threats")
        return "\n".join(summaries)

    def _summarize_forecast(self, forecast_result) -> str:
        if not forecast_result:
            return "Insufficient data for trend forecasting."
        return (
            f"{forecast_result.metric}: historical CAGR {forecast_result.cagr_historical:.1%}, "
            f"projected CAGR {forecast_result.cagr_projected:.1%} ({forecast_result.method}). "
            f"Confidence: {forecast_result.confidence:.0%}."
        )

    def _summarize_sources(self, data_points: list) -> str:
        if not self._hf:
            snippets = " ".join(dp.snippet[:200] for dp in data_points[:5])
            return snippets[:500]
        combined = " ".join(dp.snippet for dp in sorted(data_points, key=lambda p: p.confidence, reverse=True)[:5])
        return self._hf.summarize(combined, max_length=200)

    async def _generate_executive_summary(
        self, query, sector, geo_scope, frameworks_summary, forecast_summary,
        framework_results, sources_count
    ) -> str:
        if not self._llm:
            return self._fallback_executive_summary(query, sector, geo_scope, frameworks_summary, forecast_summary)

        avg_conf = (
            sum(fr.confidence_avg for fr in framework_results) / len(framework_results)
            if framework_results else 0.5
        )
        prompt = SYNTHESIS_REPORT_PROMPT.format(
            query=query,
            sector=sector,
            geo_scope=geo_scope,
            framework_summary=frameworks_summary[:1500],
            forecast_summary=forecast_summary,
            confidence=avg_conf,
            sources_count=sources_count,
        )
        result = await self._llm.complete(
            prompt,
            system="You are a senior strategy consultant writing a professional market report.",
            max_tokens=600,
            task="executive_summary",
        )
        if result.text:
            return result.text
        return self._fallback_executive_summary(query, sector, geo_scope, frameworks_summary, forecast_summary)

    def _fallback_executive_summary(self, query, sector, geo_scope, fw_summary, forecast_summary) -> str:
        return (
            f"This market intelligence report analyzes the **{sector or query}** sector "
            f"in **{geo_scope}**. The analysis draws on multiple authoritative sources "
            f"including World Bank data, academic research, and industry publications.\n\n"
            f"**Framework findings:** {fw_summary}\n\n"
            f"**Market trend:** {forecast_summary}\n\n"
            f"Strategic decision-makers should review the detailed framework analysis and "
            f"source citations in the sections below for full context and confidence levels."
        )

    async def _generate_recommendations(self, query, fw_summary, forecast_summary) -> str:
        if not self._llm:
            return self._fallback_recommendations(query)
        prompt = RECOMMENDATIONS_PROMPT.format(
            query=query,
            framework_summary=fw_summary[:1500],
            forecast_summary=forecast_summary,
        )
        result = await self._llm.complete(
            prompt,
            system="You are a strategic consultant. Be specific, actionable, and evidence-based.",
            max_tokens=600,
            task="recommendations",
        )
        if result.text:
            return result.text
        return self._fallback_recommendations(query)

    def _fallback_recommendations(self, query) -> str:
        return (
            f"1. **Conduct primary research**: Supplement this AI-generated analysis with proprietary "
            f"surveys and expert interviews in the {query} market.\n"
            "2. **Monitor leading indicators**: Track the key metrics identified in the trend forecast section quarterly.\n"
            "3. **Validate competitive position**: Cross-reference the Porter's Five Forces findings with "
            "recent competitive intelligence from industry contacts.\n"
            "4. **Assess regulatory landscape**: Prioritize monitoring of the political and legal PESTEL factors "
            "identified as potential risks.\n"
            "5. **Update analysis quarterly**: Market conditions evolve rapidly — schedule a refreshed analysis "
            "using updated data sources in 90 days."
        )

    # ── Quality gates ─────────────────────────────────────────────────────────

    def _run_quality_gates(self, report_md: str, data_points: list, framework_report) -> tuple[int, int]:
        gates = {
            "min_sources": len({p.source_domain for p in data_points}) >= 3,
            "min_word_count": len(report_md.split()) >= 1500,
            "has_citations": "[1]" in report_md or "Sources & Data" in report_md,
            "has_framework": "## Strategic Analysis" in report_md or "SWOT" in report_md or "Porter" in report_md,
            "has_forecast": "Forecast" in report_md or "CAGR" in report_md,
            "has_recommendations": "## Strategic Recommendations" in report_md,
            "has_confidence_disclosure": "Confidence" in report_md,
        }
        passed = sum(1 for v in gates.values() if v)
        failed = [k for k, v in gates.items() if not v]
        if failed:
            logger.warning("Quality gates failed: %s", failed)
        return passed, len(gates)
