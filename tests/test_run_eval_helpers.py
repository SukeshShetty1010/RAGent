"""
tests/test_run_eval_helpers.py

Unit test for evaluation/run_eval.py's _load_golden_set -- a plain JSONL
parse, no engine involved.
"""

from __future__ import annotations

import pytest

from evaluation.run_eval import _load_golden_set

pytestmark = pytest.mark.unit


def test_load_golden_set_parses_jsonl_and_skips_blank_lines(tmp_path):
    path = tmp_path / "golden_set.jsonl"
    path.write_text(
        '{"id": "1", "query": "a"}\n'
        "\n"
        '{"id": "2", "query": "b"}\n',
        encoding="utf-8",
    )

    records = _load_golden_set(path)

    assert records == [{"id": "1", "query": "a"}, {"id": "2", "query": "b"}]


def test_load_golden_set_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert _load_golden_set(path) == []
