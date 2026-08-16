"""
tests/test_usage_counter.py

Unit tests for utils/usage_counter.py's in-process accounting. Uses a
fresh UsageCounter() instance (not the .get() singleton) so tests don't
interfere with each other or with real request counts. Qdrant flush/read
is exercised in Phase 7's live verification, not here.
"""

import pytest

pytestmark = pytest.mark.unit


def test_record_accumulates_requests_and_tokens():
    from utils.usage_counter import UsageCounter

    counter = UsageCounter()
    counter.record("gemini", "chat", prompt_tokens=10, completion_tokens=5)
    counter.record("gemini", "chat", prompt_tokens=20, completion_tokens=8)

    key = next(iter(counter._counts))
    entry = counter._counts[key]
    assert entry["requests"] == 2
    assert entry["prompt_tokens"] == 30
    assert entry["completion_tokens"] == 13
    assert entry["fallback_events"] == 0


def test_record_fallback_flag():
    from utils.usage_counter import UsageCounter

    counter = UsageCounter()
    counter.record("groq", "chat", fallback=True)

    key = next(iter(counter._counts))
    assert counter._counts[key]["fallback_events"] == 1


def test_record_count_param_for_batched_calls():
    from utils.usage_counter import UsageCounter

    counter = UsageCounter()
    counter.record("gemini", "embedding", count=50)

    key = next(iter(counter._counts))
    assert counter._counts[key]["requests"] == 50


def test_flush_failure_is_fail_soft(monkeypatch):
    """A Qdrant error during flush must not raise -- it's telemetry, and
    the pending deltas must be restored so a later flush can retry."""
    from utils.usage_counter import UsageCounter

    counter = UsageCounter()
    counter.record("gemini", "chat", prompt_tokens=5, completion_tokens=2)

    monkeypatch.setenv("QDRANT_URL", "http://localhost:1")  # unreachable
    ok = counter.flush()

    assert ok is False
    key = next(iter(counter._counts))
    assert counter._counts[key]["requests"] == 1
