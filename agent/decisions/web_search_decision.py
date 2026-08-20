"""
agent/decisions/web_search_decision.py

Bounded, single-shot agentic decision: given weak/ambiguous local
retrieval quality, should the orchestrator attempt a web-search
augmentation?

This is the one deliberately-inserted "true agentic loop" decision
point in an otherwise deterministic pipeline. It replaces the
ambiguous half of `retriever/orchestrator.py`'s old boolean OR logic
(QUALITY_WEAK / temporal-signal cases) with a single LLM judgment call
over full context (query, task, quality report, evidence).

Certain cases (QUALITY_EMPTY) are NOT routed through this function —
they stay a hard deterministic `True` in the orchestrator, since there
is no ambiguity to resolve and no honesty-gate call budget to spend.

Fail-soft: any failure (network error, malformed JSON, missing/bad
keys, validation error) falls back to recreating the exact prior
deterministic boolean, tagged `source="deterministic_fallback"`, so
the system never behaves worse than before this feature existed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from agent.task_router import TaskType
from retriever.quality_gate import QualityReport, QualityStatus
from llm.ragent_client import chat_completion_decision

logger = logging.getLogger("WEB_SEARCH_DECISION")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


PROMPT_TEMPLATE = """You are a retrieval-augmentation gatekeeper for a RAG system.
Decide whether the system should attempt a WEB SEARCH to supplement local evidence.

Query: {query}
Task type: {task}
Local evidence quality: {quality_status} (reason: {quality_reason})
Local evidence relevance score (cross-encoder logit, unbounded — higher is better, roughly: <0 weak, >3 strong): {confidence_score}
Query has temporal signal (asks about recent/dated info): {has_temporal_signal}
Web fallback allowed by routing policy: {allow_web_fallback}
Top local evidence (titles + relevance scores):
{evidence_lines}

Respond with STRICT JSON only, matching this schema exactly:
{{"should_search_web": <true|false>, "reason": "<short justification, one sentence>", "confidence": <float 0.0-1.0>}}
"""


class WebSearchDecision(BaseModel):
    should_search_web: bool
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["llm", "deterministic_fallback"]


def _deterministic_fallback(
    quality_report: QualityReport,
    allow_web_fallback: bool,
) -> WebSearchDecision:
    """Recreates the original orchestrator boolean logic verbatim."""
    should = False
    if quality_report.status in {QualityStatus.QUALITY_EMPTY, QualityStatus.QUALITY_WEAK}:
        should = True
    elif allow_web_fallback and quality_report.has_temporal_signal:
        should = True

    return WebSearchDecision(
        should_search_web=should,
        reason="deterministic_fallback: LLM decision unavailable, reverted to rule-based logic",
        confidence=1.0,
        source="deterministic_fallback",
    )


def decide_web_search(
    *,
    query: str,
    task: TaskType,
    quality_report: QualityReport,
    allow_web_fallback: bool,
    evidence_summary: List[Dict[str, Any]],
) -> WebSearchDecision:
    """
    Bounded, single-shot LLM decision: should we attempt web augmentation?
    """
    try:
        evidence_lines = "\n".join(
            f"- {e.get('source_title', '?')} (score={float(e.get('score', 0.0)):.2f})"
            for e in evidence_summary
        ) or "(none)"

        prompt = PROMPT_TEMPLATE.format(
            query=query,
            task=task.value,
            quality_status=quality_report.status.value,
            quality_reason=quality_report.reason,
            confidence_score=quality_report.confidence_score,
            has_temporal_signal=quality_report.has_temporal_signal,
            allow_web_fallback=allow_web_fallback,
            evidence_lines=evidence_lines,
        )

        raw = chat_completion_decision(
            prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        parsed = json.loads(raw)

        raw_confidence = float(parsed.get("confidence", 0.5))
        clamped_confidence = max(0.0, min(1.0, raw_confidence))
        if clamped_confidence != raw_confidence:
            logger.warning(
                f"WebSearchDecision LLM returned out-of-range confidence "
                f"{raw_confidence!r}, clamped to {clamped_confidence}"
            )

        return WebSearchDecision(
            should_search_web=bool(parsed["should_search_web"]),
            reason=str(parsed.get("reason", ""))[:300],
            confidence=clamped_confidence,
            source="llm",
        )

    except Exception as exc:
        logger.warning(f"WebSearchDecision LLM call failed (fail-soft): {exc}")
        return _deterministic_fallback(quality_report, allow_web_fallback)
