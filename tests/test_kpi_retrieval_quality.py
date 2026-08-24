"""
tests/test_kpi_retrieval_quality.py

Unit tests for KPI/Retrieval_Quality_KPI.py's evidence-hit-rate / entity-
coverage aggregation. RageEngine is faked (module-local monkeypatch);
tests.evaluation_metrics.calculate_evidence_hit_rate/calculate_entity_coverage
run for real against the fake's canned evidence, so the aggregation math
itself is exercised.
"""

from __future__ import annotations

import re

import pytest

import KPI.Retrieval_Quality_KPI as retrieval_quality_kpi

pytestmark = pytest.mark.unit


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# One canned response per TRAFFIC_WITH_TRUTH query (matched by exact query
# text so this stays correct even if TRAFFIC_WITH_TRUTH's order changes).
_RESPONSES = {
    "Compare Far Cry 5 vs Assassin’s Creed Valhalla": {
        "evidence": [{"source_title": "Far Cry 5"}],  # partial coverage: 1 of 2 expected
        "quality_status": "quality_ok",
        "confidence_score": 0.9,
        "merge_state": "LOCAL_ONLY",
    },
    "Top 5 things to do in Far Cry 5": {
        "evidence": [{"source_title": "Far Cry 5"}],
        "quality_status": "quality_weak",
        "confidence_score": 0.8,
        "merge_state": "LOCAL_ONLY",
    },
    "What is the release date of Far Cry 5?": {
        "evidence": [],  # miss: no retrieved entities at all
        "quality_status": "quality_empty",
        "confidence_score": 0.5,
        "merge_state": "WEB_AUGMENTED",
    },
    "Latest patch notes for Assassin’s Creed Valhalla": {
        "evidence": [{"source_title": "Assassin’s Creed Valhalla"}],
        "quality_status": "quality_ok",
        "confidence_score": 0.7,
        "merge_state": "LOCAL_ONLY",
    },
}


class FakeEngine:
    def run(self, query: str) -> dict:
        canned = _RESPONSES[query]
        return {
            "evidence": canned["evidence"],
            "kpis": {
                "quality_status": canned["quality_status"],
                "confidence_score": canned["confidence_score"],
            },
            "agent_decisions": {"merge_state": canned["merge_state"]},
        }


def _extract_pct(out: str, label: str) -> float:
    match = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*(\d+\.\d+)%", out)
    assert match, f"could not find row {label!r} in output:\n{out}"
    return float(match.group(1))


def test_retrieval_quality_kpi_aggregate_metrics(monkeypatch, capsys):
    monkeypatch.setattr(retrieval_quality_kpi, "RageEngine", FakeEngine)

    retrieval_quality_kpi.RetrievalQualityKPI().run()

    out = _strip_ansi(capsys.readouterr().out)

    # hit on 3 of 4 queries (release-date query has empty evidence) -> 75%
    assert _extract_pct(out, "Evidence Hit Rate") == pytest.approx(75.0)
    # coverage: 0.5 + 1.0 + 0.0 + 1.0 = 2.5 / 4 -> 62.50%
    assert _extract_pct(out, "Avg Entity Coverage") == pytest.approx(62.5)
    # web fallback triggered on exactly 1 of 4 (merge_state != LOCAL_ONLY)
    assert _extract_pct(out, "Web Fallback Trigger Rate") == pytest.approx(25.0)

    conf_match = re.search(r"\|\s*Avg Retrieval Confidence\s*\|\s*([\d.]+)\s*\|", out)
    assert conf_match, out
    # (0.9 + 0.8 + 0.5 + 0.7) / 4 = 0.725
    assert float(conf_match.group(1)) == pytest.approx(0.725)
