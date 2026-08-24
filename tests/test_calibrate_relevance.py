"""
tests/test_calibrate_relevance.py

Unit tests for evaluation/calibrate_relevance.py's pure scoring helpers
against synthetic records -- no retriever, no Qdrant, no live reranker
call. (Importing the module does eagerly load the local reranker's ONNX
model per retriever/rag_retriever.py's module-level load -- same
precedent as tests/test_reranker.py, which is also marked unit.)
"""

from __future__ import annotations

import pytest

from evaluation.calibrate_relevance import (
    _f1_at_threshold,
    _floor_candidates,
    _best_split,
    _summarize,
)

pytestmark = pytest.mark.unit


def _rec(max_relevance: float, should_refuse: bool) -> dict:
    return {"max_relevance": max_relevance, "should_refuse": should_refuse}


def test_f1_at_threshold_perfect_separation():
    records = [
        _rec(-5.0, should_refuse=True),
        _rec(-3.0, should_refuse=True),
        _rec(2.0, should_refuse=False),
        _rec(4.0, should_refuse=False),
    ]
    result = _f1_at_threshold(records, threshold=0.0)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_f1_at_threshold_counts_false_positives_and_negatives():
    records = [
        _rec(-5.0, should_refuse=True),   # predicted refuse (< 0), correct -> tp
        _rec(1.0, should_refuse=True),    # predicted answer (>= 0), wrong -> fn
        _rec(-1.0, should_refuse=False),  # predicted refuse, wrong -> fp
        _rec(2.0, should_refuse=False),   # predicted answer, correct -> tn
    ]
    result = _f1_at_threshold(records, threshold=0.0)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_f1_at_threshold_handles_zero_predicted_refusals():
    """No record predicted as refuse -> precision denominator is 0."""
    records = [_rec(5.0, should_refuse=False), _rec(3.0, should_refuse=True)]
    result = _f1_at_threshold(records, threshold=-100.0)
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["precision"] == 0.0


def test_best_split_finds_the_max_f1_threshold():
    # Clean gap between -3.0 and 2.0 -- any threshold in that gap gets
    # perfect F1; _best_split should land on one of the observed values
    # (candidates are drawn from the data, not synthesized midpoints).
    records = [
        _rec(-5.0, should_refuse=True),
        _rec(-3.0, should_refuse=True),
        _rec(2.0, should_refuse=False),
        _rec(4.0, should_refuse=False),
    ]
    best = _best_split(records)
    assert best is not None
    assert best["f1"] == 1.0


def test_best_split_returns_none_for_empty_records():
    assert _best_split([]) is None


def test_floor_candidates_computes_midpoints_and_weak_pct():
    answerable = [_rec(1.0, False), _rec(2.0, False), _rec(4.0, False), _rec(4.0, False)]
    candidates = _floor_candidates(answerable)

    # Adjacent pairs: (1.0, 2.0) and (2.0, 4.0); the (4.0, 4.0) tie is
    # skipped since a zero-width gap can't be a floor.
    assert len(candidates) == 2

    first = candidates[0]
    assert first["midpoint"] == pytest.approx(1.5)
    assert first["gap"] == pytest.approx(1.0)
    # values < 1.5: just the 1.0 -> 1 of 4 = 25%
    assert first["answerable_weak_pct"] == pytest.approx(25.0)

    second = candidates[1]
    assert second["midpoint"] == pytest.approx(3.0)
    # values < 3.0: 1.0 and 2.0 -> 2 of 4 = 50%
    assert second["answerable_weak_pct"] == pytest.approx(50.0)


def test_floor_candidates_empty_when_fewer_than_two_values():
    assert _floor_candidates([_rec(1.0, False)]) == []
    assert _floor_candidates([]) == []


def test_summarize_empty_records():
    result = _summarize([])
    assert result == {"count": 0, "min": None, "max": None, "mean": None}


def test_summarize_computes_min_max_mean_at_six_decimals():
    records = [_rec(0.0000371, False), _rec(0.0000372, False)]
    result = _summarize(records)
    assert result["count"] == 2
    assert result["min"] == pytest.approx(0.000037)
    assert result["max"] == pytest.approx(0.000037)
    assert result["mean"] == pytest.approx(0.000037)
