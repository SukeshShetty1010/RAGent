"""
tests/test_engine_contract.py

Hermetic unit tests for T7 (AUDIT_TASKS.md): RageEngine and
StreamingRageEngine used to be two hand-maintained copies of the same
7-step pipeline that quietly drifted apart. RageEngine is now a thin
subclass of StreamingRageEngine (engine/execution_engine.py), and this
file is the regression suite that makes the drift it fixed impossible
to reintroduce silently:

  - both engines must report the same KPI *key set*                 (7d)
  - a failed generation must not report a latency, and a genuine
    0.0 latency must not collapse to None                            (7a)
  - a validator exception must never destroy an already-generated
    answer                                                            (7e)
  - a failed generation must emit a "failed" stage, not "completed"   (7f)
  - set_trace_attributes must actually land -- both on the normal
    path (llm_ran, output_validation) and on the cancel path
    (cancelled=True), which requires the write to happen while the
    trace is still open                                               (7i)
  - both engines fall back to the exact same static refusal object    (7c)

No network, no credentials: engines are built via object.__new__ plus
stub collaborators, following the pattern in
tests/test_streaming_cancellation.py.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pytest

from agent.capability.capability_types import AnswerCapability
from engine.contracts import INSUFFICIENT_REFUSAL
from engine.execution_engine import RageEngine
from engine.execution_engine_streaming import StreamingRageEngine
from tests.test_streaming_cancellation import _Stub, _config, _decision, _find_stage, _quality


# ============================================================
# Shared helpers (reused by tests/test_insufficient_refusal.py)
# ============================================================

def _build(cls, **overrides):
    engine = object.__new__(cls)
    engine.router = overrides.get("router") or _Stub()
    engine.strategy_selector = overrides.get("strategy_selector") or _Stub()
    engine.orchestrator = overrides.get("orchestrator") or _Stub()
    engine.capability_assessor = overrides.get("capability_assessor") or _Stub()
    engine.context_assembler = overrides.get("context_assembler") or _Stub()
    engine.prompt_manager = overrides.get("prompt_manager") or _Stub()
    engine._closed = False
    return engine


def _full_pipeline_kwargs(
    capability: AnswerCapability,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    chunks = chunks if chunks is not None else [{"source_title": "Game Wiki", "content": "text"}]
    return dict(
        router=_Stub(route=lambda query, intent_schema_version=None: _decision()),
        strategy_selector=_Stub(select=lambda decision: _config()),
        orchestrator=_Stub(
            run=lambda query, decision, config: (chunks, "vector_only", _quality(), None, _quality())
        ),
        capability_assessor=_Stub(
            assess=lambda intent_signals, evidence, quality: capability
        ),
        context_assembler=_Stub(assemble=lambda chunks, task: chunks),
        prompt_manager=_Stub(generate_prompt=lambda **kw: "prompt text"),
    )


def _fake_stream(text: str):
    """A chat_completion_streaming stand-in that yields `text` word-by-word."""

    def _gen(prompt, max_tokens=None, on_chunk=None, **kwargs):
        for word in text.split():
            piece = word + " "
            if on_chunk:
                on_chunk(piece)
            yield piece

    return _gen


def _boom_stream(prompt, max_tokens=None, on_chunk=None, **kwargs):
    raise RuntimeError("llm down")
    yield  # pragma: no cover -- makes this a generator function


class _FakeTracing:
    """Records set_trace_attributes calls, but raises if one arrives
    while trace_request's context is not open -- the exact bug 7i was."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self._active = False

    @contextmanager
    def trace_request(self, query):
        self._active = True
        try:
            yield None
        finally:
            self._active = False

    def set_trace_attributes(self, **kwargs):
        if not self._active:
            raise AssertionError("set_trace_attributes called outside an active trace")
        self.calls.append(kwargs)

    def record_generation(self, **kwargs):
        if not self._active:
            raise AssertionError("record_generation called outside an active trace")


# ============================================================
# 1. KPI key-set parity (7d)
# ============================================================

@pytest.mark.unit
def test_kpi_key_sets_match_between_engines(monkeypatch):
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming",
        _fake_stream("Answer text (Source: 'Game Wiki')."),
    )
    kwargs = _full_pipeline_kwargs(AnswerCapability.FULL)

    streaming_result = _build(StreamingRageEngine, **kwargs).run("q")
    blocking_result = _build(RageEngine, **kwargs).run("q")

    assert set(streaming_result.keys()) == set(blocking_result.keys()) == {
        "final_answer", "agent_decisions", "evidence", "kpis", "raw_metrics",
    }
    assert set(streaming_result["kpis"].keys()) == set(blocking_result["kpis"].keys())


# ============================================================
# 2. llm_latency_ms: None on failure, a genuine 0.0 survives (7a)
# ============================================================

@pytest.mark.unit
def test_llm_latency_none_on_failure(monkeypatch):
    monkeypatch.setattr("llm.ragent_client_streaming.chat_completion_streaming", _boom_stream)

    engine = _build(StreamingRageEngine, **_full_pipeline_kwargs(AnswerCapability.FULL))
    result = engine.run("q")

    assert result["kpis"]["llm_ran"] is False
    assert result["kpis"]["llm_latency_ms"] is None


@pytest.mark.unit
def test_llm_latency_zero_is_not_collapsed_to_none(monkeypatch):
    monkeypatch.setattr(
        "engine.execution_engine_streaming.time.perf_counter", lambda: 100.0
    )
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming",
        _fake_stream("Answer text (Source: 'Game Wiki')."),
    )

    engine = _build(StreamingRageEngine, **_full_pipeline_kwargs(AnswerCapability.FULL))
    result = engine.run("q")

    assert result["kpis"]["llm_ran"] is True
    assert result["kpis"]["llm_latency_ms"] == 0.0


# ============================================================
# 3. A validator exception must not destroy a good answer (7e)
# ============================================================

@pytest.mark.unit
def test_validator_exception_does_not_destroy_answer(monkeypatch):
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming",
        _fake_stream("A generated answer (Source: 'Game Wiki')."),
    )

    def _boom_validate(*args, **kwargs):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr("engine.execution_engine_streaming.validate_answer", _boom_validate)

    engine = _build(StreamingRageEngine, **_full_pipeline_kwargs(AnswerCapability.FULL))
    result = engine.run("q")

    assert result["final_answer"] == "A generated answer (Source: 'Game Wiki')."
    assert result["kpis"]["llm_ran"] is True
    assert "output_validation" not in result["agent_decisions"]


# ============================================================
# 4. Failed generation emits "failed", not "completed" (7f)
# ============================================================

@pytest.mark.unit
def test_failed_generation_emits_failed_stage(monkeypatch):
    monkeypatch.setattr("llm.ragent_client_streaming.chat_completion_streaming", _boom_stream)

    engine = _build(StreamingRageEngine, **_full_pipeline_kwargs(AnswerCapability.FULL))
    result = engine.run_streaming("q")

    assert _find_stage(result.stages, "generation", "failed") is not None
    assert _find_stage(result.stages, "generation", "completed") is None


# ============================================================
# 5. set_trace_attributes actually lands, cancel path included (7b/7i)
# ============================================================

@pytest.mark.unit
def test_trace_attributes_reach_active_trace_on_normal_path(monkeypatch):
    fake = _FakeTracing()
    monkeypatch.setattr("engine.execution_engine_streaming.tracing", fake)
    monkeypatch.setattr(
        "llm.ragent_client_streaming.chat_completion_streaming",
        _fake_stream("Answer text (Source: 'Game Wiki')."),
    )

    engine = _build(StreamingRageEngine, **_full_pipeline_kwargs(AnswerCapability.FULL))
    engine.run_streaming("q")

    final_call = fake.calls[-1]
    assert final_call["llm_ran"] is True
    assert final_call["cancelled"] is False
    assert "output_validation" in final_call


@pytest.mark.unit
def test_trace_attributes_reach_active_trace_on_cancel_path(monkeypatch):
    fake = _FakeTracing()
    monkeypatch.setattr("engine.execution_engine_streaming.tracing", fake)

    engine = _build(StreamingRageEngine)  # default boom-stubs; nothing should run
    cancel_event = threading.Event()
    cancel_event.set()
    engine.run_streaming("q", cancel_event=cancel_event)

    final_call = fake.calls[-1]
    assert final_call["cancelled"] is True
    assert final_call["llm_ran"] is False


# ============================================================
# 6. Both engines fall back to the exact same refusal object (7c)
# ============================================================

@pytest.mark.unit
def test_both_engines_use_the_same_static_refusal_constant(monkeypatch):
    monkeypatch.setattr("llm.ragent_client_streaming.chat_completion_streaming", _boom_stream)

    kwargs = _full_pipeline_kwargs(AnswerCapability.INSUFFICIENT)
    streaming_answer = _build(StreamingRageEngine, **kwargs).run("q")["final_answer"]
    blocking_answer = _build(RageEngine, **kwargs).run("q")["final_answer"]

    assert streaming_answer == INSUFFICIENT_REFUSAL
    assert blocking_answer == INSUFFICIENT_REFUSAL
