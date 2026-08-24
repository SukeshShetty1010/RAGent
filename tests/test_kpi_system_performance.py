"""
tests/test_kpi_system_performance.py

Unit tests for KPI/System_Performance_KPI.py. This module is the one KPI
runner with two independent RageEngine construction sites: its own direct
call, and (via tests.regression_suite.RegressionRunner ->
tests.evaluation_runner.EvaluationRunner) a second, separately-imported
RageEngine used to replay the regression vault -- both module-local names
need patching, or the second construction reaches the real engine.

FakeEngine answers regression-vault queries by looking up the matching
tests.regression_suite.REGRESSION_VAULT case and returning exactly what
that case expects, so the regression pass count is a real, computed 3/3
rather than a hardcoded fixture -- and Context Efficiency / Latency
Attribution get seeded through MetricsRegistry's public API on every call
so both engine-construction sites contribute to the same aggregation.
"""

from __future__ import annotations

import re

import pytest

import KPI.System_Performance_KPI as system_performance_kpi
import tests.evaluation_runner as evaluation_runner
from tests.regression_suite import REGRESSION_VAULT
from utils.observability import MetricsRegistry

pytestmark = pytest.mark.unit


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


_CASES_BY_QUERY = {case.test_case.query: case.test_case for case in REGRESSION_VAULT}

# Hand-written answers, one per known bug_id, each satisfying that case's
# required_structure_pattern -- the query text itself doesn't necessarily
# match its own pattern (e.g. BUG-001's comparison query contains none of
# "Gameplay|Story|World Design|Tone|Systems"), so this can't be derived
# generically. If regression_suite.py's REGRESSION_VAULT patterns change,
# this map needs a matching update.
_ANSWERS_BY_BUG_ID = {
    "BUG-001": "Gameplay comparison: both titles differ in tone and systems.",
    "BUG-002": "Top 5 things to do: 1. Explore the map.",
    "BUG-003": "This is the latest update available.",
}
_BUG_ID_BY_QUERY = {case.test_case.query: case.bug_id for case in REGRESSION_VAULT}


class FakeEngine:
    def run(self, query: str) -> dict:
        registry = MetricsRegistry.get()
        registry.observe("context_input_chunks", 20)
        registry.observe("context_final_chunks", 8)
        registry.observe("latency::REQUEST_TOTAL -> retrieval", 50.0)
        registry.observe("latency::REQUEST_TOTAL -> generation", 150.0)

        case = _CASES_BY_QUERY.get(query)
        if case is None:
            # Not a regression-vault query (e.g. System_Performance_KPI's
            # own representative-request call) -- content doesn't matter,
            # it's never compared against an expectation.
            return {
                "final_answer": "stub answer",
                "evidence": [],
                "kpis": {"answer_capability": "full"},
                "agent_decisions": {"task": "FACTUAL", "answer_capability": "full"},
                "raw_metrics": {},
            }

        # Build a response that satisfies exactly this case's contract so
        # the regression pass count is a real, computed 3/3.
        bug_id = _BUG_ID_BY_QUERY[query]
        answer = _ANSWERS_BY_BUG_ID[bug_id]
        assert case.required_structure_pattern.search(answer), (
            f"fixture bug: canned answer for {bug_id} doesn't match its pattern"
        )
        return {
            "final_answer": answer,
            "evidence": [{"source_title": t} for t in case.expected_source_titles],
            "kpis": {"answer_capability": case.expected_capability.value},
            "agent_decisions": {
                "task": case.expected_task.value,
                "answer_capability": case.expected_capability.value,
            },
            "raw_metrics": {},
        }

    def close(self) -> None:
        pass


def test_system_performance_kpi_regression_and_latency_attribution(monkeypatch, capsys):
    monkeypatch.setattr(system_performance_kpi, "RageEngine", FakeEngine)
    monkeypatch.setattr(evaluation_runner, "RageEngine", FakeEngine)

    system_performance_kpi.SystemPerformanceKPI().run()

    out = _strip_ansi(capsys.readouterr().out)

    total = len(REGRESSION_VAULT)
    assert f"{total}/{total}" in out

    # retrieval=50ms, generation=150ms -> 200 total -> 25% / 75%
    assert "retrieval: 25.00%" in out
    assert "generation: 75.00%" in out

    # 4 engine.run() calls total (1 direct + 3 regression cases), each
    # seeding input=20/final=8 -> 80 -> 32, ratio 0.6
    assert "60.00%" in out
    assert "Input chunks: 80, Final chunks used: 32" in out
