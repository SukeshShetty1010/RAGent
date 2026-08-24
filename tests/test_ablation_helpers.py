"""
tests/test_ablation_helpers.py

Unit tests for evaluation/ablation.py's pure helpers (_load_jsonl,
_markdown_table, _build_output) against synthetic summaries -- no
retriever, no RAGAS, no live judge calls.
"""

from __future__ import annotations

import json

import pytest

from evaluation.ablation import MODES, _load_jsonl, _markdown_table, _build_output

pytestmark = pytest.mark.unit


def _retrieval_summary(**overrides) -> dict:
    base = {
        mode: {"mean_precision_at_k": 0.5, "mean_entity_coverage": 0.6, "n": 10}
        for mode in MODES
    }
    for mode, values in overrides.items():
        base[mode].update(values)
    return base


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")

    records = _load_jsonl(path)

    assert records == [{"a": 1}, {"a": 2}]


def test_markdown_table_without_ragas_summary():
    table = _markdown_table(_retrieval_summary(), None)

    assert "| Mode | Precision@K | Entity Coverage |" in table
    assert "RAGAS Context Precision" not in table
    for mode in MODES:
        assert f"`{mode}`" in table


def test_markdown_table_with_ragas_summary_and_missing_mode():
    ragas_summary = {"hybrid_rerank": {"mean_context_precision": 0.7}}
    table = _markdown_table(_retrieval_summary(), ragas_summary)

    assert "RAGAS Context Precision" in table
    assert "0.7000" in table
    # bm25/dense/hybrid have no RAGAS score in this summary -> "n/a"
    assert "n/a" in table


def test_build_output_hybrid_rerank_wins():
    retrieval_summary = _retrieval_summary(hybrid_rerank={"mean_precision_at_k": 0.9})
    output = _build_output(retrieval_summary, None, judge_backend=None)

    assert output["hybrid_rerank_wins_on_precision_at_k"] is True


def test_build_output_hybrid_rerank_loses_to_a_higher_scoring_mode():
    """Pins the exact bug class this comparison must catch: a non-default
    mode outscoring hybrid_rerank on precision@k must be reported as a
    loss, not silently flagged as a win."""
    retrieval_summary = _retrieval_summary(
        hybrid_rerank={"mean_precision_at_k": 0.5},
        bm25={"mean_precision_at_k": 0.8},
    )
    output = _build_output(retrieval_summary, None, judge_backend=None)

    assert output["hybrid_rerank_wins_on_precision_at_k"] is False


def test_build_output_ragas_complete_requires_every_mode():
    retrieval_summary = _retrieval_summary()
    partial_ragas = {m: {"mean_context_precision": 0.5} for m in MODES[:-1]}

    output = _build_output(retrieval_summary, partial_ragas, judge_backend="groq")
    assert output["ragas_complete"] is False

    full_ragas = {m: {"mean_context_precision": 0.5} for m in MODES}
    output = _build_output(retrieval_summary, full_ragas, judge_backend="groq")
    assert output["ragas_complete"] is True
