"""
tests/test_kpi_faith_fair.py

Unit tests for KPI/Faith_Fair_KPI.py's capability accounting and citation-
grounding rate. RageEngine is faked (module-local monkeypatch); the real
tests.evaluation_metrics.calculate_grounding_fidelity does the citation
matching against the fake's canned final_answer/evidence, so the
grounded-rate math itself is exercised for real, not mocked away.
"""

from __future__ import annotations

import re

import pytest

import KPI.Faith_Fair_KPI as faith_fair_kpi

pytestmark = pytest.mark.unit

# TRAFFIC has 6 queries; cycling full/partial/insufficient every 3 calls
# gives 2 of each, and two capability-specific answer shapes so the
# citation-grounding math has real signal instead of an all-or-nothing
# result.
_CYCLE = ["full", "partial", "insufficient"]


class FakeEngine:
    def __init__(self) -> None:
        self._calls = 0

    def run(self, query: str) -> dict:
        capability = _CYCLE[self._calls % 3]
        self._calls += 1

        if capability == "full":
            answer = "This is a fully cited claim (Source: 'Far Cry 5')."
            evidence = [{"source_title": "Far Cry 5"}]
        elif capability == "partial":
            answer = "This is an uncited claim."
            evidence = []
        else:
            answer = ""
            evidence = []

        return {
            "final_answer": answer,
            "evidence": evidence,
            "kpis": {"answer_capability": capability},
        }

    def close(self) -> None:
        pass


def _extract_pct(out: str, label: str) -> float:
    match = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*(\d+\.\d+)%", out)
    assert match, f"could not find row {label!r} in output:\n{out}"
    return float(match.group(1))


def test_faith_fair_kpi_capability_and_grounding_rates(monkeypatch, capsys):
    monkeypatch.setattr(faith_fair_kpi, "RageEngine", FakeEngine)

    faith_fair_kpi.FaithFairKPI().run()

    out = capsys.readouterr().out

    # honest_answers = full(2) + partial(2) = 4 of 6 -> 66.67%
    assert _extract_pct(out, "Answered Rate (Full+Partial)") == pytest.approx(66.67)
    # graceful_degradations = partial(2) of 6 -> 33.33%
    assert _extract_pct(out, "Graceful Degradation Coverage") == pytest.approx(33.33)
    # grounded: 2 full-capability sentences cited, 2 partial-capability
    # sentences uncited, insufficient contributes no sentences -> 2/4 = 50%
    assert _extract_pct(out, "Citation-Grounded Sentence Rate") == pytest.approx(50.0)
    assert _extract_pct(out, "Uncited Claim Rate") == pytest.approx(50.0)


def test_faith_fair_kpi_closes_engine(monkeypatch):
    closed = []
    fake = FakeEngine()
    fake.close = lambda: closed.append(True)
    monkeypatch.setattr(faith_fair_kpi, "RageEngine", lambda: fake)

    faith_fair_kpi.FaithFairKPI().run()

    assert closed == [True]
