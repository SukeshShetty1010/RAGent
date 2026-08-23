"""
tests/test_cost_latency_metrics.py

Unit tests for evaluation/cost_latency_metrics.py's pure aggregation
function against synthetic run records — no engine, no Qdrant, no LLM
calls.

The cost cases pin AUDIT_TASKS.md §17's finding: a Gemini-served query
costs exactly 0.0 (llm/pricing.py prices the free tier at zero on
purpose), so aggregating with a truthiness filter drops every free query
and reports the mean over only the Groq fallbacks — which is what made
runs_2026-08-21_default.jsonl read as $0.000106/query when 49 of its 50
queries cost nothing.
"""

from __future__ import annotations

import pytest

from evaluation.cost_latency_metrics import compute_cost_latency_metrics

pytestmark = pytest.mark.unit


def _rec(**extra):
    record = {
        "id": "x",
        "engine_latency_ms": 1000.0,
        "llm_latency_ms": 500.0,
        "prompt_tokens": 800,
        "completion_tokens": 100,
        "cost_usd": 0.0,
    }
    record.update(extra)
    return record


def test_zero_cost_queries_count_toward_the_per_query_mean():
    """49 free + 1 paid must average over all 50, not over the paid one."""
    records = [_rec() for _ in range(49)] + [_rec(cost_usd=0.000106)]

    metrics = compute_cost_latency_metrics(records)
    cost = metrics["cost_usd_per_query"]

    assert cost["queries_priced"] == 50
    assert cost["queries_at_zero_cost"] == 49
    assert cost["total_for_run"] == pytest.approx(0.000106)
    assert cost["mean"] == pytest.approx(0.000106 / 50, rel=1e-3)
    assert cost["median"] == 0.0


def test_never_generated_query_is_excluded_rather_than_counted_as_free():
    """A hard refusal records None, not 0.0 — it is not a free generation."""
    records = [_rec(cost_usd=None, prompt_tokens=None, completion_tokens=None), _rec()]

    metrics = compute_cost_latency_metrics(records)

    assert metrics["cost_usd_per_query"]["queries_priced"] == 1
    assert metrics["cost_usd_per_query"]["queries_at_zero_cost"] == 1
    assert metrics["tokens_per_query"]["mean_prompt_tokens"] == 800


def test_errored_records_are_dropped_before_aggregation():
    records = [_rec(), _rec(error="boom", cost_usd=99.0)]

    metrics = compute_cost_latency_metrics(records)

    assert metrics["total_scored"] == 1
    assert metrics["total_errored"] == 1
    assert metrics["cost_usd_per_query"]["total_for_run"] == 0.0


def test_stage_attribution_is_a_percentage_of_request_total():
    records = [
        _rec(stage_latency_ms={"REQUEST_TOTAL": 1000.0, "Retrieval": 800.0}),
        _rec(stage_latency_ms={"REQUEST_TOTAL": 1000.0, "Retrieval": 600.0}),
    ]

    metrics = compute_cost_latency_metrics(records)
    retrieval = metrics["stage_latency_attribution"]["Retrieval"]

    assert retrieval["avg_ms"] == pytest.approx(700.0)
    assert retrieval["pct_of_total"] == pytest.approx(70.0)
