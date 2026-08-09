"""
tests/test_refusal_metrics.py

Unit tests for evaluation/refusal_metrics.py's pure scoring function
against synthetic run records — no engine, no Qdrant, no LLM calls.
"""

from __future__ import annotations

import pytest

from evaluation.refusal_metrics import compute_refusal_metrics


def _rec(should_refuse: bool, refused: bool, merge_state: str = "LOCAL_ONLY", **extra):
    return {
        "id": extra.get("id", "x"),
        "query": extra.get("query", "q"),
        "should_refuse": should_refuse,
        "answer_capability": "insufficient" if refused else "full",
        "merge_state": merge_state,
    }


def test_perfect_scoring_gives_precision_recall_of_one():
    records = [
        _rec(should_refuse=True, refused=True),
        _rec(should_refuse=False, refused=False),
    ]
    metrics = compute_refusal_metrics(records)
    assert metrics["refusal_precision"] == 1.0
    assert metrics["refusal_recall"] == 1.0
    assert metrics["refusal_f1"] == 1.0
    assert metrics["false_answer_rate"] == 0.0
    assert metrics["over_refusal_rate"] == 0.0


def test_false_answer_is_should_refuse_but_not_refused():
    # Should have refused (unanswerable) but the engine answered anyway.
    records = [
        _rec(should_refuse=True, refused=False, id="u1", query="unanswerable"),
        _rec(should_refuse=False, refused=False),
    ]
    metrics = compute_refusal_metrics(records)
    assert metrics["confusion"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 1}
    assert metrics["false_answer_rate"] == 1.0
    assert metrics["false_answers"] == [{"id": "u1", "query": "unanswerable"}]
    # No positive predictions at all -> precision defined as 0.0, not divide-by-zero.
    assert metrics["refusal_precision"] == 0.0
    assert metrics["refusal_recall"] == 0.0


def test_over_refusal_is_answerable_but_refused():
    records = [
        _rec(should_refuse=False, refused=True, id="a1", query="answerable"),
        _rec(should_refuse=True, refused=True),
    ]
    metrics = compute_refusal_metrics(records)
    assert metrics["confusion"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 0}
    assert metrics["over_refusal_rate"] == 1.0
    assert metrics["over_refusals"] == [{"id": "a1", "query": "answerable"}]
    assert metrics["refusal_precision"] == 0.5
    assert metrics["refusal_recall"] == 1.0


def test_not_tautological_recall_can_be_below_one():
    # This is the property calculate_honesty_rate cannot have: it is
    # graded against an external label and can genuinely score badly.
    records = [
        _rec(should_refuse=True, refused=False),
        _rec(should_refuse=True, refused=False),
        _rec(should_refuse=True, refused=True),
    ]
    metrics = compute_refusal_metrics(records)
    assert metrics["refusal_recall"] == pytest.approx(1 / 3, rel=1e-3)


def test_errored_records_excluded_from_scoring_but_counted():
    records = [
        _rec(should_refuse=True, refused=True),
        {"id": "e1", "query": "boom", "error": "engine crashed"},
    ]
    metrics = compute_refusal_metrics(records)
    assert metrics["total_scored"] == 1
    assert metrics["total_errored"] == 1
    assert metrics["errored_ids"] == ["e1"]


def test_breakdown_by_merge_state():
    records = [
        _rec(should_refuse=True, refused=True, merge_state="LOCAL_WEB_ATTEMPTED"),
        _rec(should_refuse=False, refused=False, merge_state="LOCAL_ONLY"),
        _rec(should_refuse=False, refused=False, merge_state="LOCAL_PLUS_WEB"),
    ]
    metrics = compute_refusal_metrics(records)
    assert metrics["by_merge_state"]["LOCAL_WEB_ATTEMPTED"] == {
        "total": 1, "refused": 1, "should_refuse": 1,
    }
    assert metrics["by_merge_state"]["LOCAL_ONLY"] == {
        "total": 1, "refused": 0, "should_refuse": 0,
    }
