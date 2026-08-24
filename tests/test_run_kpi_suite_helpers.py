"""
tests/test_run_kpi_suite_helpers.py

Unit tests for evaluation/run_kpi_suite.py's pure helpers: _strip_ansi,
_run_module (a generic call-and-capture harness), and
_derive_overall_status (the hard-crash-vs-partial-failure classification
that mirrors what python -m KPI.Unified_KPI_Runner would actually do).
No KPI classes or engine involved -- these are exercised directly with
synthetic callables/module-result dicts.
"""

from __future__ import annotations

import pytest

from evaluation.run_kpi_suite import _strip_ansi, _run_module, _derive_overall_status

pytestmark = pytest.mark.unit


def test_strip_ansi_removes_escape_codes():
    colored = "\x1b[92mgreen\x1b[0m and \x1b[1mbold\x1b[0m"
    assert _strip_ansi(colored) == "green and bold"


def test_strip_ansi_leaves_plain_text_untouched():
    assert _strip_ansi("no color here") == "no color here"


def test_run_module_captures_return_value_and_stdout():
    result = _run_module("ok_module", caught_by_real_runner=True, call=lambda: (print("hi"), 42)[1])

    assert result["status"] == "completed"
    assert result["structured"] == 42
    assert result["stdout"] == "hi\n"
    assert result["exception_type"] is None
    assert result["duration_s"] >= 0.0


def test_run_module_isolates_an_exception():
    def _boom():
        print("before crash")
        raise ValueError("kaboom")

    result = _run_module("bad_module", caught_by_real_runner=False, call=_boom)

    assert result["status"] == "crashed"
    assert result["exception_type"] == "ValueError"
    assert result["exception_message"] == "kaboom"
    assert "before crash" in result["stdout"]
    assert "ValueError" in result["traceback"]


def test_run_module_strips_ansi_from_captured_stdout():
    result = _run_module("colored", caught_by_real_runner=True, call=lambda: print("\x1b[92mgreen\x1b[0m"))
    assert result["stdout"] == "green\n"


def _mod(status: str, caught: bool) -> dict:
    return {"status": status, "caught_by_real_runner": caught}


def test_derive_overall_status_all_completed():
    modules = [_mod("completed", True), _mod("completed", False)]
    assert _derive_overall_status(modules) == "all_completed"


def test_derive_overall_status_partial_failure_when_crash_is_caught_by_real_runner():
    modules = [_mod("completed", False), _mod("crashed", True), _mod("completed", False)]
    assert _derive_overall_status(modules) == "partial_failure"


def test_derive_overall_status_hard_crash_when_uncaught_module_fails():
    modules = [_mod("completed", True), _mod("crashed", False), _mod("completed", True)]
    assert _derive_overall_status(modules) == "hard_crash_before_completion"


def test_derive_overall_status_hard_crash_takes_priority_over_partial_failure():
    """An earlier caught crash plus a later uncaught crash is still a
    hard crash overall -- the real entrypoint would have died at the
    uncaught one regardless of what failed before it."""
    modules = [_mod("crashed", True), _mod("crashed", False)]
    assert _derive_overall_status(modules) == "hard_crash_before_completion"
