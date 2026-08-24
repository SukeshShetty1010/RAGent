"""
tests/test_kpi_context_engineering.py

Unit tests for KPI/Context_Engineering_KPI.py's aggregation math. The
RageEngine construction and execution is faked (module-local monkeypatch,
matching tests/test_llm_clients.py:37's pattern) since KPI/ has no
dependency-injection seam -- but the metrics themselves are real:
FakeEngine.run() seeds MetricsRegistry through its public API
(observe/inc/record, see utils/observability.py) with fixed synthetic
values, and this test asserts the KPI module's printed percentages match
hand-computed values from those seeds.
"""

from __future__ import annotations

import pytest

import KPI.Context_Engineering_KPI as context_engineering_kpi
from utils.observability import MetricsRegistry

pytestmark = pytest.mark.unit


class FakeEngine:
    def __init__(self) -> None:
        self.registry = MetricsRegistry.get()

    def run(self, query: str) -> dict:
        self.registry.observe("context_input_chunks", 20)
        self.registry.observe("context_final_chunks", 8)
        self.registry.inc("context_redundant_rejections", 2)
        self.registry.observe("context_deduped_chunks", 10)
        self.registry.record("prompt_budget_mode", "concise")
        return {
            "final_answer": "stub answer",
            "evidence": [],
            "kpis": {"answer_capability": "full"},
            "agent_decisions": {"task": "FACTUAL"},
        }


def test_context_engineering_kpi_prints_hand_computed_percentages(monkeypatch, capsys):
    monkeypatch.setattr(context_engineering_kpi, "RageEngine", FakeEngine)

    context_engineering_kpi.ContextEngineeringKPI().run()

    out = capsys.readouterr().out

    # 5 TRAFFIC queries * (input=20, final=8) -> 100 -> 40, ratio 0.6
    assert "60.00%" in out
    assert "100 → 40" in out

    # 5 * inc(2) = 10 rejections over 5 * observe(10) = 50 candidates -> 20%
    assert "20.00%" in out

    # every call records "prompt_budget_mode"="concise", no "insufficient"
    # prompt_mode entries -> 5/5 compliant
    assert "100.00%" in out


def test_context_engineering_kpi_resets_registry_between_runs(monkeypatch, capsys):
    """MetricsRegistry is thread-local, not run-scoped -- .run() must
    reset() first or a second run would double-count against the first."""
    monkeypatch.setattr(context_engineering_kpi, "RageEngine", FakeEngine)

    kpi = context_engineering_kpi.ContextEngineeringKPI()
    kpi.run()
    capsys.readouterr()  # discard first run's output
    kpi.run()

    out = capsys.readouterr().out
    # Same synthetic seeds every run -> same percentages, not doubled.
    assert "60.00%" in out
    assert "100 → 40" in out
