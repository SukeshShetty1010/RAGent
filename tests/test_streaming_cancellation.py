"""
tests/test_streaming_cancellation.py

Hermetic unit tests for T19 (client-disconnect cancellation) and the SSE
polling loop in api/main.py that makes it possible.

StreamingRageEngine.__init__ builds a Qdrant-backed RetrievalOrchestrator,
so every engine-level test here constructs the engine via object.__new__
and injects stub collaborators instead -- no network, no credentials.
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Optional

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from agent.capability.capability_types import AnswerCapability
from agent.task_router import RouterDecision, TaskType
from engine.execution_engine_streaming import (
    StreamingRageEngine,
    StreamingResult,
    StreamingStage,
)
from retriever.quality_gate import QualityReport, QualityStatus
from retriever.strategy_selector import RetrievalConfiguration


# ============================================================
# Engine-level fixtures/helpers
# ============================================================

def _boom(name: str):
    def _raise(*args, **kwargs):
        raise AssertionError(f"{name} should not have been called")
    return _raise


def _boom_error(message: str):
    def _raise(*args, **kwargs):
        raise RuntimeError(message)
    return _raise


class _Stub:
    """Attribute bag whose collaborator methods raise unless overridden.

    A test that forgets to stub a method it actually needs gets a loud
    AssertionError instead of a silent AttributeError-turned-fatal-error,
    and a test that asserts a step was skipped gets that assertion for
    free -- the boom fires if the pipeline reaches it.
    """

    def __init__(self, **methods):
        for attr in ("route", "select", "run", "assess", "assemble", "generate_prompt"):
            setattr(self, attr, _boom(attr))
        for name, fn in methods.items():
            setattr(self, name, fn)


def _decision() -> RouterDecision:
    return RouterDecision(task=TaskType.FACTUAL, intent_signals=set(), reason="test")


def _config() -> RetrievalConfiguration:
    return RetrievalConfiguration(
        limit=5,
        use_window_expansion=False,
        use_query_decomposition=False,
        allow_web_fallback=False,
    )


def _quality() -> QualityReport:
    return QualityReport(
        status=QualityStatus.QUALITY_OK,
        reason="ok",
        confidence_score=0.9,
        has_temporal_signal=False,
    )


def _build_engine(
    *,
    router=None,
    strategy_selector=None,
    orchestrator=None,
    capability_assessor=None,
    context_assembler=None,
    prompt_manager=None,
) -> StreamingRageEngine:
    engine = object.__new__(StreamingRageEngine)
    engine.router = router or _Stub()
    engine.strategy_selector = strategy_selector or _Stub()
    engine.orchestrator = orchestrator or _Stub()
    engine.capability_assessor = capability_assessor or _Stub()
    engine.context_assembler = context_assembler or _Stub()
    engine.prompt_manager = prompt_manager or _Stub()
    engine._closed = False
    return engine


def _find_stage(stages: List[StreamingStage], name: str, status: Optional[str] = None):
    for s in stages:
        if s.name == name and (status is None or s.status == status):
            return s
    return None


# ============================================================
# 1. cancel_event already set before run_streaming starts
# ============================================================

@pytest.mark.unit
def test_cancel_before_start_short_circuits_pipeline():
    """Every collaborator stays a booby-trapped boom -- none should fire."""
    engine = _build_engine()
    cancel_event = threading.Event()
    cancel_event.set()

    result = engine.run_streaming("query", cancel_event=cancel_event)

    assert result.kpis["cancelled"] is True
    assert result.kpis["task_success"] is False
    assert result.kpis["llm_ran"] is False
    assert _find_stage(result.stages, "routing", "cancelled") is not None


# ============================================================
# 2. Disconnect noticed mid-retrieval (the actual §19 scenario --
#    retrieval is the ~106s call, everything after it is checkpointed)
# ============================================================

@pytest.mark.unit
def test_cancel_mid_retrieval_skips_generation():
    cancel_event = threading.Event()

    def _orchestrator_run(query, decision, config):
        # Simulate a disconnect landing while retrieval is in flight.
        cancel_event.set()
        return [], "vector_only", _quality(), None, _quality()

    engine = _build_engine(
        router=_Stub(route=lambda query, intent_schema_version=None: _decision()),
        strategy_selector=_Stub(select=lambda decision: _config()),
        orchestrator=_Stub(run=_orchestrator_run),
        # capability_assessor / context_assembler / prompt_manager stay
        # default boom-stubs: none of them should be reached once the
        # next checkpoint (before STEP 4) sees the event set.
    )

    result = engine.run_streaming("query", cancel_event=cancel_event)

    assert result.kpis["cancelled"] is True
    assert _find_stage(result.stages, "retrieval", "completed") is not None
    assert _find_stage(result.stages, "capability", "cancelled") is not None
    assert _find_stage(result.stages, "prompt_construction", "started") is None


# ============================================================
# 3. No cancel_event at all -- the default every CLI/KPI caller uses
# ============================================================

@pytest.mark.unit
def test_no_cancel_event_runs_full_pipeline():
    engine = _build_engine(
        router=_Stub(route=lambda query, intent_schema_version=None: _decision()),
        strategy_selector=_Stub(select=lambda decision: _config()),
        orchestrator=_Stub(
            run=lambda query, decision, config: ([], "vector_only", _quality(), None, _quality())
        ),
        capability_assessor=_Stub(
            assess=lambda intent_signals, evidence, quality: AnswerCapability.INSUFFICIENT
        ),
        context_assembler=_Stub(assemble=lambda chunks, task: chunks),
        # Prompt construction runs unconditionally (capability only gates
        # generation), so this must be a working stub, not a boom.
        prompt_manager=_Stub(generate_prompt=lambda **kw: "prompt text"),
    )

    result = engine.run_streaming("query")  # cancel_event defaults to None

    assert result.kpis["cancelled"] is False
    assert _find_stage(result.stages, "generation", "skipped") is not None


# ============================================================
# 4. A genuine error must not be misclassified as a cancellation
# ============================================================

@pytest.mark.unit
def test_genuine_error_is_fatal_not_cancelled():
    engine = _build_engine(
        router=_Stub(route=_boom_error("routing exploded")),
    )

    result = engine.run_streaming("query", cancel_event=threading.Event())

    assert result.kpis["cancelled"] is False
    assert result.final_answer == "An internal error occurred while processing the request."
    assert _find_stage(result.stages, "error", "failed") is not None


# ============================================================
# 5. API-level: the polling event_generator forwards stage + done frames
# ============================================================

class _FakeStreamingEngine:
    """Stands in for get_engine() -- emits canned stages, then returns."""

    def __init__(self, stages: List[StreamingStage], sleep_before_stages: float = 0.0):
        self._stages = stages
        self._sleep_before_stages = sleep_before_stages

    def run_streaming(self, query, on_token_callback=None, on_stage_callback=None,
                       cancel_event=None, options=None):
        if self._sleep_before_stages:
            time.sleep(self._sleep_before_stages)
        for stage in self._stages:
            if on_stage_callback:
                on_stage_callback(stage)
        return StreamingResult(
            final_answer="hello",
            agent_decisions={},
            evidence=[],
            kpis={"cancelled": False},
            raw_metrics={},
            stages=self._stages,
        )


@pytest.mark.unit
def test_chat_endpoint_streams_stage_then_done_events(monkeypatch):
    stages = [
        StreamingStage(name="routing", status="started", data={}, duration_ms=None),
        StreamingStage(name="routing", status="completed", data={"task": "factual"}, duration_ms=1.2),
    ]
    monkeypatch.setattr(api_main, "get_engine", lambda: _FakeStreamingEngine(stages))

    client = TestClient(api_main.app)
    response = client.post("/api/chat", json={"query": "hi"})

    assert response.status_code == 200
    body = response.text
    stage_index = body.find("event: stage")
    done_index = body.find("event: done")
    assert stage_index != -1
    assert done_index != -1
    assert stage_index < done_index


@pytest.mark.unit
def test_chat_endpoint_survives_slow_stage_emission(monkeypatch):
    """A stage that arrives only after the 1s poll interval elapses must
    not be dropped and must not make the loop spin -- proves the
    queue.Empty -> continue path in event_generator works."""
    stages = [
        StreamingStage(name="retrieval", status="started", data={}, duration_ms=None),
        StreamingStage(name="retrieval", status="completed", data={}, duration_ms=1200.0),
    ]
    monkeypatch.setattr(
        api_main, "get_engine",
        lambda: _FakeStreamingEngine(stages, sleep_before_stages=1.2),
    )

    client = TestClient(api_main.app)
    response = client.post("/api/chat", json={"query": "hi"})

    assert response.status_code == 200
    assert response.text.count("event: stage") == 2
    assert "event: done" in response.text
