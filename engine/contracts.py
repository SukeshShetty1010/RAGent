# ============================================================
# engine/contracts.py
# Shared answer constants and result shapes for both engines.
#
# Single source of truth for what RageEngine and StreamingRageEngine
# must agree on byte-for-byte (T7): the fixed answer strings, the
# 5-key result contract tests/verify_engine.py enforces, and the
# QualityReport -> dict shape used in agent_decisions.
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from retriever.quality_gate import QualityReport

INSUFFICIENT_REFUSAL = (
    "I don't have enough reliable information "
    "to answer this request safely."
)

GENERATION_FAILED = "I could not generate a response at this time."

INTERNAL_ERROR = "An internal error occurred while processing the request."


class ExecutionResult(TypedDict):
    final_answer: str
    agent_decisions: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    kpis: Dict[str, Any]
    raw_metrics: Dict[str, Any]


def quality_report_dict(q: QualityReport) -> Dict[str, Any]:
    return {
        "status": q.status.value,
        "confidence_score": q.confidence_score,
        "has_temporal_signal": q.has_temporal_signal,
        "max_relevance": q.max_relevance,
        "entity_grounded": q.entity_grounded,
        "evidence_count": q.evidence_count,
    }
