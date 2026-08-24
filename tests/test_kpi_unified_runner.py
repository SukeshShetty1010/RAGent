"""
tests/test_kpi_unified_runner.py

Unit tests for KPI/Unified_KPI_Runner.py's fail-soft orchestration. Each
of the 5 underlying KPI classes is faked (module-local monkeypatch,
patched on KPI.Unified_KPI_Runner since that's where they're imported by
name) so this test controls exactly which one raises.

KPI/Unified_KPI_Runner.py wraps FaithFairKPI, RetrievalQualityKPI, and
SystemPerformanceKPI's .run() in try/except, but NOT ContextEngineeringKPI
(line 70) or ResumeKPIDashboard (lines 112-114) -- this asymmetry is
confirmed here rather than assumed to be a bug, per the module's current
behavior.
"""

from __future__ import annotations

import pytest

import KPI.Unified_KPI_Runner as unified_runner

pytestmark = pytest.mark.unit


def _make_fake_kpi(sentinel: str, should_raise: bool = False):
    class FakeKPI:
        def run(self) -> None:
            if should_raise:
                raise RuntimeError(f"{sentinel} boom")
            print(sentinel)

    return FakeKPI


class FakeIntentDashboard:
    """Stands in for ResumeKPIDashboard, which UnifiedDashboard calls as
    `ResumeKPIDashboard().run()` then `ResumeKPIDashboard.print_dashboard(...)`
    (a staticmethod call on the class itself, not the instance)."""

    def __init__(self, should_raise: bool = False) -> None:
        self._should_raise = should_raise

    def run(self):
        if self._should_raise:
            raise RuntimeError("intent boom")
        print("intent-ran")
        return ["metric"]

    @staticmethod
    def print_dashboard(metrics) -> None:
        print(f"intent-dashboard:{metrics}")


def _patch_all_succeed(monkeypatch):
    monkeypatch.setattr(unified_runner, "ContextEngineeringKPI", _make_fake_kpi("context-ran"))
    monkeypatch.setattr(unified_runner, "FaithFairKPI", _make_fake_kpi("faith-ran"))
    monkeypatch.setattr(unified_runner, "RetrievalQualityKPI", _make_fake_kpi("retrieval-ran"))
    monkeypatch.setattr(unified_runner, "SystemPerformanceKPI", _make_fake_kpi("system-ran"))
    monkeypatch.setattr(unified_runner, "ResumeKPIDashboard", FakeIntentDashboard)


def test_unified_dashboard_runs_all_five_when_none_fail(monkeypatch, capsys):
    _patch_all_succeed(monkeypatch)

    unified_runner.UnifiedDashboard().run()

    out = capsys.readouterr().out
    assert "context-ran" in out
    assert "faith-ran" in out
    assert "retrieval-ran" in out
    assert "system-ran" in out
    assert "intent-ran" in out
    assert "intent-dashboard:['metric']" in out


@pytest.mark.parametrize("wrapped_attr", ["FaithFairKPI", "RetrievalQualityKPI", "SystemPerformanceKPI"])
def test_unified_dashboard_swallows_failure_in_wrapped_modules(monkeypatch, capsys, wrapped_attr):
    _patch_all_succeed(monkeypatch)
    monkeypatch.setattr(unified_runner, wrapped_attr, _make_fake_kpi("irrelevant", should_raise=True))

    # Must not raise -- these three are explicitly try/except-wrapped.
    unified_runner.UnifiedDashboard().run()

    out = capsys.readouterr().out
    assert "failed safely" in out
    # Execution continues past the failure to the remaining modules.
    assert "intent-dashboard" in out


def test_unified_dashboard_propagates_failure_in_context_engineering(monkeypatch):
    """ContextEngineeringKPI is NOT try/except-wrapped (line 70) -- a
    failure here crashes the whole dashboard run, unlike the three
    modules above."""
    _patch_all_succeed(monkeypatch)
    monkeypatch.setattr(unified_runner, "ContextEngineeringKPI", _make_fake_kpi("irrelevant", should_raise=True))

    with pytest.raises(RuntimeError, match="boom"):
        unified_runner.UnifiedDashboard().run()


def test_unified_dashboard_propagates_failure_in_intent_dashboard(monkeypatch):
    """ResumeKPIDashboard is NOT try/except-wrapped (lines 112-114)
    either -- same asymmetry as ContextEngineeringKPI."""
    _patch_all_succeed(monkeypatch)
    monkeypatch.setattr(unified_runner, "ResumeKPIDashboard", lambda: FakeIntentDashboard(should_raise=True))

    with pytest.raises(RuntimeError, match="intent boom"):
        unified_runner.UnifiedDashboard().run()
