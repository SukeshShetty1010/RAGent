"""
tests/test_kpi_intent_agent_control.py

Unit tests for KPI/Intent_Agent_Control.py's routing-accuracy and
routing-determinism computation. RageEngine is faked (module-local
monkeypatch) with a per-query, per-pass task map so both the "correct vs
expected" and "stable across two passes" branches get real signal instead
of an all-pass fixture.
"""

from __future__ import annotations

import pytest

import KPI.Intent_Agent_Control as intent_agent_control

pytestmark = pytest.mark.unit

# (pass_a_task, pass_b_task) per query, keyed by the exact TRAFFIC query
# text. Chosen so routing accuracy (vs expected) and determinism (A vs B)
# land on different, hand-computable percentages:
#   - idx0: correct + stable
#   - idx1: incorrect + unstable
#   - idx2: correct + unstable
#   - idx3: incorrect + unstable
#   - idx4: correct + stable
# -> correct_routes = 3/5 = 60.00%, stable_decisions = 2/5 = 40.00%
_TASKS_BY_QUERY = {
    "Compare Far Cry 5 vs Assassin’s Creed Valhalla": ("comparison", "comparison"),
    "Top 5 things to do in Far Cry 5": ("factual", "listicle"),
    "What is the release date of Far Cry 5?": ("factual", "open"),
    "Explain why Far Cry 5 is controversial": ("comparison", "listicle"),
    "Latest update for Assassin’s Creed Valhalla": ("open", "open"),
}


class FakeEngine:
    def __init__(self) -> None:
        self._pass_counts: dict[str, int] = {}

    def run(self, query: str) -> dict:
        pass_index = self._pass_counts.get(query, 0)
        self._pass_counts[query] = pass_index + 1
        task = _TASKS_BY_QUERY[query][pass_index]
        return {"agent_decisions": {"task": task}}


def test_intent_agent_control_accuracy_and_determinism(monkeypatch):
    monkeypatch.setattr(intent_agent_control, "RageEngine", FakeEngine)

    dashboard = intent_agent_control.ResumeKPIDashboard()
    metrics = dashboard.run()

    by_name = {m.name: m for m in metrics}

    assert by_name["Task Routing Accuracy"].value == pytest.approx(60.0)
    assert by_name["Task Routing Accuracy"].samples == 5
    assert by_name["Intent Signal Accuracy"].value == pytest.approx(60.0)
    assert by_name["Routing Determinism Rate"].value == pytest.approx(40.0)
    assert by_name["Routing Determinism Rate"].samples == 5


def test_intent_agent_control_print_dashboard_does_not_raise(monkeypatch, capsys):
    monkeypatch.setattr(intent_agent_control, "RageEngine", FakeEngine)

    dashboard = intent_agent_control.ResumeKPIDashboard()
    metrics = dashboard.run()
    intent_agent_control.ResumeKPIDashboard.print_dashboard(metrics)

    out = capsys.readouterr().out
    assert "Routing Determinism Rate" in out
