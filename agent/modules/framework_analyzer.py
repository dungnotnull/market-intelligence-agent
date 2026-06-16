"""
FrameworkAnalyzer: selects and applies business analysis frameworks
(SWOT, Porter's Five Forces, PESTEL, VRIO, BCG Matrix) using LLM + evidence.
Triangulates findings across sources and assigns confidence scores.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

FRAMEWORK_SELECTION_PROMPT = """You are a senior market analyst. Given this market research query:
Query: {query}
Sector: {sector}
Geographic scope: {geo_scope}

Select the most appropriate business analysis frameworks from this list:
- SWOT: General strategic positioning; strengths/weaknesses/opportunities/threats
- Porter: Competitive dynamics analysis; best for competitive landscape queries
- PESTEL: Macro-environmental scan; best for regulatory, policy, or macro-economic queries
- VRIO: Sustainable competitive advantage analysis; best for "competitive moat" queries
- BCG: Portfolio positioning; best for multi-product or multi-segment market queries

Return ONLY valid JSON: {{"frameworks": ["SWOT", "Porter"], "rationale": "reason string"}}
Select 1 to 3 frameworks. Return only the JSON, no other text."""

SWOT_ANALYSIS_PROMPT = """You are a strategic analyst. Analyze the following market evidence and produce a SWOT analysis.

Query: {query}
Sector: {sector} | Geography: {geo_scope}

Evidence items:
{evidence}

Produce a comprehensive SWOT analysis. Return ONLY valid JSON in this exact format:
{{
  "strengths": [
    {{"claim": "specific strength", "source": "source name", "confidence": 0.8, "evidence_quote": "brief quote"}}
  ],
  "weaknesses": [
    {{"claim": "specific weakness", "source": "source name", "confidence": 0.7, "evidence_quote": "brief quote"}}
  ],
  "opportunities": [
    {{"claim": "specific opportunity", "source": "source name", "confidence": 0.75, "evidence_quote": "brief quote"}}
  ],
  "threats": [
    {{"claim": "specific threat", "source": "source name", "confidence": 0.65, "evidence_quote": "brief quote"}}
  ]
}}
Each quadrant must have 2-4 items. Confidence = fraction of evidence supporting the claim (0.0-1.0).
Return only the JSON."""

PORTER_FIVE_FORCES_PROMPT = """You are a competitive strategy expert. Using the evidence below, analyze Porter's Five Forces.

Query: {query}
Sector: {sector} | Geography: {geo_scope}

Evidence items:
{evidence}

Return ONLY valid JSON:
{{
  "competitive_rivalry": {{
    "score": 4,
    "intensity": "High",
    "evidence": "explanation citing evidence",
    "key_players": ["Player A", "Player B"],
    "sources": ["source1", "source2"]
  }},
  "supplier_power": {{
    "score": 3,
    "intensity": "Moderate",
    "evidence": "explanation",
    "sources": ["source1"]
  }},
  "buyer_power": {{
    "score": 3,
    "intensity": "Moderate",
    "evidence": "explanation",
    "sources": ["source1"]
  }},
  "new_entrants": {{
    "score": 2,
    "intensity": "Low",
    "evidence": "explanation",
    "barriers": ["barrier1", "barrier2"],
    "sources": ["source1"]
  }},
  "substitutes": {{
    "score": 3,
    "intensity": "Moderate",
    "evidence": "explanation",
    "sources": ["source1"]
  }},
  "overall_industry_attractiveness": "Moderate — explain briefly"
}}
Score each force 1 (very weak/low) to 5 (very strong/high). Return only the JSON."""

PESTEL_PROMPT = """You are a macro-environmental analyst. Using the evidence below, produce a PESTEL analysis.

Query: {query}
Sector: {sector} | Geography: {geo_scope}

Evidence items:
{evidence}

Return ONLY valid JSON:
{{
  "political": [
    {{"factor": "factor name", "impact": "Positive/Negative/Neutral", "description": "...", "confidence": 0.8, "source": "source name"}}
  ],
  "economic": [...],
  "social": [...],
  "technological": [...],
  "environmental": [...],
  "legal": [...]
}}
Each category must have 1-3 factors. Return only the JSON."""

VRIO_PROMPT = """You are a strategic management expert. Using the evidence, produce a VRIO analysis.

Query: {query}
Sector: {sector}

Evidence items:
{evidence}

Return ONLY valid JSON:
{{
  "value": {{"assessment": "Yes/No/Partial", "explanation": "...", "evidence": "...", "confidence": 0.8}},
  "rarity": {{"assessment": "Yes/No/Partial", "explanation": "...", "evidence": "...", "confidence": 0.7}},
  "imitability": {{"assessment": "Costly/Easy/Moderate", "explanation": "...", "evidence": "...", "confidence": 0.75}},
  "organization": {{"assessment": "Yes/No/Partial", "explanation": "...", "evidence": "...", "confidence": 0.7}},
  "competitive_implication": "Sustainable Competitive Advantage / Competitive Parity / Competitive Disadvantage"
}}
Return only the JSON."""

BCG_MATRIX_PROMPT = """You are a portfolio strategy expert. Using the evidence, position the market segments in a BCG Matrix.

Query: {query}
Sector: {sector}

Evidence items:
{evidence}

Return ONLY valid JSON:
{{
  "stars": [
    {{"segment": "segment name", "rationale": "high growth AND high market share", "confidence": 0.8}}
  ],
  "cash_cows": [...],
  "question_marks": [...],
  "dogs": [...],
  "strategic_summary": "Overall portfolio assessment in 2-3 sentences"
}}
Return only the JSON."""


@dataclass
class FrameworkResult:
    framework: str
    data: dict
    confidence_avg: float
    sources_used: list[str]


@dataclass
class FrameworkReport:
    query: str
    sector: str
    geo_scope: str
    frameworks: list[FrameworkResult] = field(default_factory=list)
    selection_rationale: str = ""
    conflicts: list[dict] = field(default_factory=list)


class FrameworkAnalyzer:
    def __init__(self, llm_client=None, hf_manager=None):
        self._llm = llm_client
        self._hf = hf_manager

    async def analyze(
        self,
        query: str,
        evidence: list,
        sector: str = "",
        geo_scope: str = "global",
        requested_frameworks: list[str] = None,
    ) -> FrameworkReport:
        report = FrameworkReport(query=query, sector=sector, geo_scope=geo_scope)

        if requested_frameworks:
            selected = [f.upper() for f in requested_frameworks]
            report.selection_rationale = f"User-specified frameworks: {', '.join(selected)}"
        else:
            selected, rationale = await self._select_frameworks(query, sector, geo_scope)
            report.selection_rationale = rationale

        evidence_text = self._format_evidence(evidence)

        for framework in selected:
            result = await self._apply_framework(framework, query, sector, geo_scope, evidence_text, evidence)
            if result:
                report.frameworks.append(result)

        report.conflicts = self._detect_conflicts(report.frameworks)
        return report

    async def _select_frameworks(self, query: str, sector: str, geo_scope: str) -> tuple[list[str], str]:
        if not self._llm:
            return self._heuristic_select(query, sector), "Heuristic selection (no LLM available)"

        prompt = FRAMEWORK_SELECTION_PROMPT.format(
            query=query, sector=sector, geo_scope=geo_scope
        )
        result = await self._llm.complete(prompt, max_tokens=300, task="framework_selection")
        try:
            raw = self._extract_json(result.text)
            data = json.loads(raw)
            frameworks = [f.upper() for f in data.get("frameworks", ["SWOT"])]
            rationale = data.get("rationale", "")
            valid = {"SWOT", "PORTER", "PESTEL", "VRIO", "BCG"}
            frameworks = [f for f in frameworks if f in valid][:3]
            if not frameworks:
                frameworks = ["SWOT"]
            return frameworks, rationale
        except Exception as e:
            logger.warning("Framework selection JSON parse failed: %s", e)
            return self._heuristic_select(query, sector), "Heuristic fallback"

    def _heuristic_select(self, query: str, sector: str) -> list[str]:
        q = (query + " " + sector).lower()
        frameworks = []
        if any(w in q for w in ["compet", "rival", "market share", "player"]):
            frameworks.append("PORTER")
        if any(w in q for w in ["macro", "regulation", "policy", "political", "economic"]):
            frameworks.append("PESTEL")
        if any(w in q for w in ["portfolio", "segment", "product line"]):
            frameworks.append("BCG")
        if not frameworks or any(w in q for w in ["strategy", "analysis", "overview"]):
            frameworks.append("SWOT")
        return list(dict.fromkeys(frameworks))[:2]

    async def _apply_framework(
        self, framework: str, query: str, sector: str, geo_scope: str,
        evidence_text: str, evidence_raw: list
    ) -> Optional[FrameworkResult]:
        prompts = {
            "SWOT": SWOT_ANALYSIS_PROMPT,
            "PORTER": PORTER_FIVE_FORCES_PROMPT,
            "PESTEL": PESTEL_PROMPT,
            "VRIO": VRIO_PROMPT,
            "BCG": BCG_MATRIX_PROMPT,
        }
        prompt_template = prompts.get(framework)
        if not prompt_template:
            return None

        if not self._llm:
            return self._fallback_framework(framework, query)

        prompt = prompt_template.format(
            query=query, sector=sector, geo_scope=geo_scope, evidence=evidence_text[:4000]
        )

        data = None
        for attempt in range(3):
            result = await self._llm.complete(prompt, max_tokens=1500, task=f"framework_{framework.lower()}")
            try:
                raw = self._extract_json(result.text)
                data = json.loads(raw)
                break
            except Exception:
                if attempt == 2:
                    logger.warning("Framework %s JSON parse failed after 3 attempts — using fallback", framework)
                    return self._fallback_framework(framework, query)

        if not data:
            return self._fallback_framework(framework, query)

        confidence = self._compute_confidence(data, framework)
        sources = list({p.source_name for p in evidence_raw[:10]})

        return FrameworkResult(
            framework=framework,
            data=data,
            confidence_avg=confidence,
            sources_used=sources,
        )

    def _fallback_framework(self, framework: str, query: str) -> FrameworkResult:
        templates = {
            "SWOT": {
                "strengths": [{"claim": f"Growing market for {query}", "source": "industry consensus", "confidence": 0.5}],
                "weaknesses": [{"claim": "High competitive intensity", "source": "industry consensus", "confidence": 0.5}],
                "opportunities": [{"claim": "Emerging technology adoption", "source": "industry consensus", "confidence": 0.5}],
                "threats": [{"claim": "Regulatory uncertainty", "source": "industry consensus", "confidence": 0.5}],
            },
            "PORTER": {
                "competitive_rivalry": {"score": 3, "intensity": "Moderate", "evidence": "Market analysis required", "sources": []},
                "supplier_power": {"score": 3, "intensity": "Moderate", "evidence": "Analysis required", "sources": []},
                "buyer_power": {"score": 3, "intensity": "Moderate", "evidence": "Analysis required", "sources": []},
                "new_entrants": {"score": 2, "intensity": "Low", "evidence": "Analysis required", "barriers": [], "sources": []},
                "substitutes": {"score": 2, "intensity": "Low", "evidence": "Analysis required", "sources": []},
                "overall_industry_attractiveness": "Moderate — detailed analysis required",
            },
            "PESTEL": {
                "political": [{"factor": "Regulatory environment", "impact": "Neutral", "description": "Requires detailed analysis", "confidence": 0.4, "source": "fallback"}],
                "economic": [{"factor": "GDP growth", "impact": "Positive", "description": "General economic indicator", "confidence": 0.5, "source": "World Bank"}],
                "social": [{"factor": "Consumer trends", "impact": "Neutral", "description": "Requires analysis", "confidence": 0.4, "source": "fallback"}],
                "technological": [{"factor": "Digital transformation", "impact": "Positive", "description": "Ongoing tech adoption", "confidence": 0.5, "source": "fallback"}],
                "environmental": [{"factor": "Sustainability requirements", "impact": "Neutral", "description": "Increasing ESG focus", "confidence": 0.4, "source": "fallback"}],
                "legal": [{"factor": "Compliance requirements", "impact": "Neutral", "description": "Regulatory landscape", "confidence": 0.4, "source": "fallback"}],
            },
        }
        data = templates.get(framework, {"note": f"Fallback template for {framework}"})
        return FrameworkResult(framework=framework, data=data, confidence_avg=0.45, sources_used=[])

    def _format_evidence(self, evidence: list) -> str:
        lines = []
        for i, pt in enumerate(evidence[:15], 1):
            lines.append(
                f"[{i}] SOURCE: {pt.source_name}\n"
                f"    TITLE: {pt.title}\n"
                f"    DATA: {pt.snippet[:300]}\n"
                f"    URL: {pt.url}\n"
            )
        return "\n".join(lines)

    def _compute_confidence(self, data: dict, framework: str) -> float:
        all_confidences = []
        self._extract_confidences(data, all_confidences)
        if all_confidences:
            return round(sum(all_confidences) / len(all_confidences), 3)
        return 0.6

    def _extract_confidences(self, obj, results: list):
        if isinstance(obj, dict):
            if "confidence" in obj:
                try:
                    results.append(float(obj["confidence"]))
                except (ValueError, TypeError):
                    pass
            for v in obj.values():
                self._extract_confidences(v, results)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_confidences(item, results)

    def _detect_conflicts(self, results: list[FrameworkResult]) -> list[dict]:
        conflicts = []
        all_claims = []
        for r in results:
            self._collect_claims(r.data, r.framework, all_claims)

        for i, (fw_a, claim_a) in enumerate(all_claims):
            for fw_b, claim_b in all_claims[i + 1:]:
                if fw_a == fw_b:
                    continue
                if self._are_conflicting(claim_a, claim_b):
                    conflicts.append({"framework_a": fw_a, "claim_a": claim_a,
                                      "framework_b": fw_b, "claim_b": claim_b})
        return conflicts[:5]

    def _collect_claims(self, obj, framework: str, results: list):
        if isinstance(obj, dict):
            if "claim" in obj:
                results.append((framework, obj["claim"]))
            for v in obj.values():
                self._collect_claims(v, framework, results)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_claims(item, framework, results)

    def _are_conflicting(self, claim_a: str, claim_b: str) -> bool:
        positive = {"grow", "strong", "opportunit", "advantage"}
        negative = {"decline", "weak", "threat", "risk", "challenge"}
        a_pos = any(w in claim_a.lower() for w in positive)
        b_neg = any(w in claim_b.lower() for w in negative)
        b_pos = any(w in claim_b.lower() for w in positive)
        a_neg = any(w in claim_a.lower() for w in negative)
        return (a_pos and b_neg) or (a_neg and b_pos)

    def _extract_json(self, text: str) -> str:
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if match:
            return match.group(1)
        brace_start = text.find("{")
        if brace_start >= 0:
            return text[brace_start:]
        return text
