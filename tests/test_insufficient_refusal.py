"""
tests/test_insufficient_refusal.py

Hermetic unit tests for T12 (AUDIT_TASKS.md): `insufficient_prompt()`
used to be built, have two metrics recorded about it, and then get
discarded in favor of a fixed refusal string. The STEP 7 `else:`
branch in engine/execution_engine_streaming.py now sends that prompt
to the LLM and gates the result through `agent.output_validator.
is_refusal()` before it ships, falling back to the static
INSUFFICIENT_REFUSAL constant on any failure.

No network, no credentials: chat_completion_streaming is monkeypatched
at its import site; engines are built via the same object.__new__ +
stub-collaborator pattern as tests/test_streaming_cancellation.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agent.capability.capability_assessor import CapabilityAssessor
from agent.capability.capability_types import AnswerCapability
from agent.intent.intent_signals import IntentSignal
from agent.task_router import RouterDecision, TaskType
from engine.contracts import INSUFFICIENT_REFUSAL
from engine.execution_engine_streaming import StreamingRageEngine
from retriever.quality_gate import QualityReport, QualityStatus
from tests.test_engine_contract import _boom_stream, _build, _fake_stream, _full_pipeline_kwargs
from tests.test_streaming_cancellation import _Stub, _config, _decision, _quality


# ============================================================
# 1. A generated refusal that passes is_refusal() ships as-is
# ============================================================

@pytest.mark.unit
def test_generated_refusal_accepted_and_reaches_final_answer(monkeypatch):
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming",
        _fake_stream("I can't answer that from the retrieved evidence about Foo."),
    )

    engine = _build(StreamingRageEngine, **_full_pipeline_kwargs(AnswerCapability.INSUFFICIENT))
    result = engine.run("q")

    assert result["final_answer"] == "I can't answer that from the retrieved evidence about Foo."
    assert result["kpis"]["refusal_mode"] == "generated"
    assert result["kpis"]["llm_ran"] is False
    assert result["kpis"]["task_success"] is False


# ============================================================
# 2. Failure modes fall back to the static constant
# ============================================================

@pytest.mark.unit
def test_empty_generated_output_falls_back_to_static_refusal(monkeypatch):
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming", _fake_stream("   ")
    )

    engine = _build(StreamingRageEngine, **_full_pipeline_kwargs(AnswerCapability.INSUFFICIENT))
    result = engine.run("q")

    assert result["final_answer"] == INSUFFICIENT_REFUSAL
    assert result["kpis"]["refusal_mode"] == "static_fallback"
    assert result["kpis"]["llm_ran"] is False
    assert result["kpis"]["task_success"] is False


@pytest.mark.unit
def test_citation_bearing_refusal_rejected_falls_back(monkeypatch):
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming",
        _fake_stream("Not enough info, but see (Source: 'Some Wiki')."),
    )

    engine = _build(StreamingRageEngine, **_full_pipeline_kwargs(AnswerCapability.INSUFFICIENT))
    result = engine.run("q")

    assert result["final_answer"] == INSUFFICIENT_REFUSAL
    assert result["kpis"]["refusal_mode"] == "static_fallback"
    assert result["kpis"]["llm_ran"] is False
    assert result["kpis"]["task_success"] is False


@pytest.mark.unit
def test_generation_exception_falls_back_to_static_refusal(monkeypatch):
    monkeypatch.setattr("llm.ragent_client_streaming.chat_completion_streaming", _boom_stream)

    engine = _build(StreamingRageEngine, **_full_pipeline_kwargs(AnswerCapability.INSUFFICIENT))
    result = engine.run("q")

    assert result["final_answer"] == INSUFFICIENT_REFUSAL
    assert result["kpis"]["refusal_mode"] == "static_fallback"
    assert result["kpis"]["llm_ran"] is False
    assert result["kpis"]["task_success"] is False


# ============================================================
# 3. capability_reason propagates into the prompt for all three
#    INSUFFICIENT causes CapabilityAssessor can produce
# ============================================================

class _PromptSpy:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def generate_prompt(self, **kwargs):
        self.calls.append(kwargs)
        return "prompt text"


def _decision_with_signals(signals):
    return RouterDecision(task=TaskType.FACTUAL, intent_signals=signals, reason="test")


@pytest.mark.unit
def test_capability_reason_no_evidence(monkeypatch):
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming", _fake_stream("refusal text")
    )
    spy = _PromptSpy()
    engine = _build(
        StreamingRageEngine,
        router=_Stub(route=lambda query, intent_schema_version=None: _decision()),
        strategy_selector=_Stub(select=lambda decision: _config()),
        orchestrator=_Stub(
            run=lambda query, decision, config: ([], "vector_only", _quality(), None, _quality())
        ),
        capability_assessor=CapabilityAssessor(),
        context_assembler=_Stub(assemble=lambda chunks, task: chunks),
        prompt_manager=spy,
    )
    engine.run("q")

    assert spy.calls[-1]["capability_reason"] == "no_evidence"


@pytest.mark.unit
def test_capability_reason_quality_empty(monkeypatch):
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming", _fake_stream("refusal text")
    )
    empty_quality = QualityReport(
        status=QualityStatus.QUALITY_EMPTY,
        reason="no_matches",
        confidence_score=0.0,
        has_temporal_signal=False,
    )
    spy = _PromptSpy()
    engine = _build(
        StreamingRageEngine,
        router=_Stub(route=lambda query, intent_schema_version=None: _decision()),
        strategy_selector=_Stub(select=lambda decision: _config()),
        orchestrator=_Stub(
            run=lambda query, decision, config: (
                [{"source_title": "X"}], "vector_only", empty_quality, None, empty_quality,
            )
        ),
        capability_assessor=CapabilityAssessor(),
        context_assembler=_Stub(assemble=lambda chunks, task: chunks),
        prompt_manager=spy,
    )
    engine.run("q")

    assert spy.calls[-1]["capability_reason"] == "quality_empty:no_matches"


@pytest.mark.unit
def test_capability_reason_comparison_entity_coverage(monkeypatch):
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming", _fake_stream("refusal text")
    )
    single_entity_chunks = [{"source_title": "Only Game", "content": "x"}]
    spy = _PromptSpy()
    engine = _build(
        StreamingRageEngine,
        router=_Stub(
            route=lambda query, intent_schema_version=None: _decision_with_signals(
                {IntentSignal.COMPARISON}
            )
        ),
        strategy_selector=_Stub(select=lambda decision: _config()),
        orchestrator=_Stub(
            run=lambda query, decision, config: (
                single_entity_chunks, "vector_only", _quality(), None, _quality(),
            )
        ),
        capability_assessor=CapabilityAssessor(),
        context_assembler=_Stub(assemble=lambda chunks, task: chunks),
        prompt_manager=spy,
    )
    engine.run("q")

    assert spy.calls[-1]["capability_reason"] == "comparison_entity_coverage"
