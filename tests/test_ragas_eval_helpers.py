"""
tests/test_ragas_eval_helpers.py

Unit tests for evaluation/ragas_eval.py's pure/file-based helpers:
_load_jsonl, _latest_default_run, _run_tag, _load_checkpoint,
_split_scored_and_failed. No RAGAS evaluate() call, no judge LLM, no
Qdrant -- all synthetic files/dicts.
"""

from __future__ import annotations

import pytest

import evaluation.ragas_eval as ragas_eval
from evaluation.ragas_eval import (
    _load_jsonl,
    _latest_default_run,
    _run_tag,
    _load_checkpoint,
    _split_scored_and_failed,
)

pytestmark = pytest.mark.unit


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
    assert _load_jsonl(path) == [{"id": "a"}, {"id": "b"}]


def test_latest_default_run_picks_lexically_last_match(monkeypatch, tmp_path):
    monkeypatch.setattr(ragas_eval, "RESULTS_DIR", tmp_path)

    (tmp_path / "runs_2026-08-01_default.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "runs_2026-08-23_default.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "runs_2026-08-10_corpusonly.jsonl").write_text("", encoding="utf-8")  # not "_default"

    latest = _latest_default_run()
    assert latest.name == "runs_2026-08-23_default.jsonl"


def test_latest_default_run_returns_none_when_no_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(ragas_eval, "RESULTS_DIR", tmp_path)
    assert _latest_default_run() is None


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("runs_2026-08-23_default", "2026-08-23_default"),
        ("runs_2026-08-23_corpusonly", "2026-08-23_corpusonly"),
        ("runs_2026-08-23", "2026-08-23"),
        ("not_a_runs_file", "not_a_runs_file"),
    ],
)
def test_run_tag_extracts_date_and_suffix(stem, expected, tmp_path):
    path = tmp_path / f"{stem}.jsonl"
    assert _run_tag(path) == expected


def test_load_checkpoint_returns_empty_dict_when_file_missing(tmp_path):
    assert _load_checkpoint(tmp_path / "does_not_exist.jsonl") == {}


def test_load_checkpoint_indexes_by_id(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text(
        '{"id": "q1", "context_precision": 0.5}\n'
        '{"id": "q2", "context_precision": 0.8}\n',
        encoding="utf-8",
    )
    checkpoint = _load_checkpoint(path)
    assert set(checkpoint.keys()) == {"q1", "q2"}
    assert checkpoint["q1"]["context_precision"] == 0.5


def test_split_scored_and_failed_separates_all_null_rows():
    scored_row = {"id": "q1", "context_precision": 0.5, "faithfulness": None, "answer_relevancy": None}
    failed_row = {"id": "q2", "context_precision": None, "faithfulness": None, "answer_relevancy": None}

    scored, failed = _split_scored_and_failed([scored_row, failed_row])

    assert scored == [scored_row]
    assert failed == [failed_row]


def test_split_scored_and_failed_empty_input():
    assert _split_scored_and_failed([]) == ([], [])
