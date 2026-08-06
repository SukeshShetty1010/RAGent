import os
import pytest

from agent.task_router import TaskType
from retriever.quality_gate import QualityReport, QualityStatus
from agent.decisions.web_search_decision import decide_web_search, WebSearchDecision
import agent.decisions.web_search_decision as web_search_decision_mod


def test_decide_web_search_live_llm_quality_weak():
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")

    report = QualityReport(
        status=QualityStatus.QUALITY_WEAK,
        reason="Only noise content detected",
        confidence_score=0.2,
        has_temporal_signal=False,
    )

    result = decide_web_search(
        query="What happened in the latest Far Cry 5 update?",
        task=TaskType.FACTUAL,
        quality_report=report,
        allow_web_fallback=True,
        evidence_summary=[{"source_title": "Far Cry 5 Store Page", "score": 0.31}],
    )

    assert isinstance(result, WebSearchDecision)
    assert isinstance(result.should_search_web, bool)
    assert result.source == "llm"
    assert 0.0 <= result.confidence <= 1.0
    assert result.reason


@pytest.mark.parametrize(
    "broken_return",
    [
        None,  # raises inside the monkeypatched function
        "not json",
        '{"foo": true}',
    ],
)
def test_decide_web_search_fallback_on_failure(monkeypatch, broken_return):
    if broken_return is None:
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated Groq outage")
        monkeypatch.setattr(web_search_decision_mod, "chat_completion_decision", _boom)
    else:
        monkeypatch.setattr(
            web_search_decision_mod,
            "chat_completion_decision",
            lambda *args, **kwargs: broken_return,
        )

    report = QualityReport(
        status=QualityStatus.QUALITY_WEAK,
        reason="Only noise content detected",
        confidence_score=0.2,
        has_temporal_signal=False,
    )

    result = decide_web_search(
        query="anything",
        task=TaskType.FACTUAL,
        quality_report=report,
        allow_web_fallback=True,
        evidence_summary=[],
    )

    assert result.source == "deterministic_fallback"
    assert result.should_search_web is True  # matches old deterministic logic for QUALITY_WEAK
