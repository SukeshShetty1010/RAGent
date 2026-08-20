import os
import pytest

from agent.task_router import TaskType
from retriever.quality_gate import QualityReport, QualityStatus
from agent.decisions.web_search_decision import decide_web_search, WebSearchDecision
import agent.decisions.web_search_decision as web_search_decision_mod


@pytest.mark.live
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


@pytest.mark.unit
def test_decide_web_search_does_not_override_max_tokens_default(monkeypatch):
    """
    Regression (AUDIT_TASKS T4): this call site used to pass
    max_tokens=150, shadowing chat_completion_decision's deliberate
    320 default and starving the Groq fallback (a reasoning model
    whose hidden reasoning tokens are billed against max_tokens) of
    output budget before it could emit any JSON.
    """
    captured_kwargs = {}

    def _capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return '{"should_search_web": true, "reason": "test", "confidence": 0.5}'

    monkeypatch.setattr(web_search_decision_mod, "chat_completion_decision", _capture)

    report = QualityReport(
        status=QualityStatus.QUALITY_WEAK,
        reason="Only noise content detected",
        confidence_score=0.2,
        has_temporal_signal=False,
    )

    decide_web_search(
        query="anything",
        task=TaskType.FACTUAL,
        quality_report=report,
        allow_web_fallback=True,
        evidence_summary=[],
    )

    assert "max_tokens" not in captured_kwargs


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize("bad_confidence", [7.88, -2.0, 1.0001])
def test_decide_web_search_clamps_out_of_range_confidence(monkeypatch, bad_confidence):
    # Regression test: an unbounded cross-encoder-scale confidence leaking into
    # the LLM's own JSON output used to raise a pydantic ValidationError here,
    # silently degrading every call to the deterministic fallback since
    # 2026-08-12. Out-of-range values must be clamped, not treated as a failure.
    monkeypatch.setattr(
        web_search_decision_mod,
        "chat_completion_decision",
        lambda *args, **kwargs: (
            '{"should_search_web": true, "reason": "test", '
            f'"confidence": {bad_confidence}}}'
        ),
    )

    report = QualityReport(
        status=QualityStatus.QUALITY_WEAK,
        reason="Only noise content detected",
        confidence_score=7.88,  # unbounded cross-encoder logit scale
        has_temporal_signal=False,
    )

    result = decide_web_search(
        query="anything",
        task=TaskType.FACTUAL,
        quality_report=report,
        allow_web_fallback=True,
        evidence_summary=[],
    )

    assert result.source == "llm"
    assert 0.0 <= result.confidence <= 1.0
