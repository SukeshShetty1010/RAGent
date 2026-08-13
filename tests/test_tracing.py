"""
tests/test_tracing.py

Hermetic tests for utils/tracing.py (Langfuse trace layer) and the
generic ProfileBlock span-hook mechanism in utils/observability.py.

No network, no real Langfuse credentials. utils/tracing.py caches its
client construction in module-level globals, so every test resets that
cache via monkeypatch to stay isolated from both the surrounding
process's real .env (never loaded here) and from other tests.
"""

from __future__ import annotations

import pytest

from llm.pricing import estimate_cost
from utils import observability, tracing

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_tracing_state(monkeypatch):
    """No LANGFUSE_* env vars, no cached client, from a clean slate every test."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_client_init_attempted", False)
    tracing._local.active = False
    tracing._local.root = None
    yield
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_client_init_attempted", False)
    tracing._local.active = False
    tracing._local.root = None


# ============================================================
# llm/pricing.py
# ============================================================

def test_estimate_cost_known_model():
    cost = estimate_cost("llama-3.1-8b-instant", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(0.05 + 0.08)


def test_estimate_cost_zero_tokens():
    assert estimate_cost("llama-3.1-8b-instant", 0, 0) == 0.0


def test_estimate_cost_unknown_model_returns_zero():
    assert estimate_cost("some-model-not-in-the-table", 1000, 1000) == 0.0


# ============================================================
# utils/tracing.py — hard no-op without both Langfuse keys
# ============================================================

def test_get_client_none_without_keys():
    assert tracing.get_client() is None


def test_get_client_none_with_only_public_key(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    assert tracing.get_client() is None


def test_trace_request_yields_none_and_is_noop():
    with tracing.trace_request("a query") as root:
        assert root is None
        assert tracing._is_active() is False  # no-op path never flips the flag

    assert tracing._is_active() is False


def test_set_trace_attributes_noop_without_active_trace():
    # Must not raise even though no trace is open.
    tracing.set_trace_attributes(task="factual", answer_capability="full")


def test_record_generation_noop_without_active_trace():
    tracing.record_generation(
        model="llama-3.1-8b-instant",
        prompt="hi",
        output="hello",
        prompt_tokens=5,
        completion_tokens=5,
        cost_usd=0.0,
        latency_ms=10.0,
    )


def test_flush_noop_without_client():
    tracing.flush()  # must not raise


# ============================================================
# utils/observability.py — generic span-hook mechanism
# ============================================================

@pytest.fixture
def _restore_hooks():
    yield
    observability.set_span_hooks(None, None)


def test_profileblock_invokes_registered_hooks(_restore_hooks):
    entered = []
    exited = []

    def enter_hook(name):
        entered.append(name)
        return f"token-for-{name}"

    def exit_hook(name, token):
        exited.append((name, token))

    observability.set_span_hooks(enter_hook, exit_hook)

    with observability.ProfileBlock("OuterStep"):
        with observability.ProfileBlock("InnerStep"):
            pass

    assert entered == ["OuterStep", "InnerStep"]
    # Inner closes before outer.
    assert exited == [
        ("InnerStep", "token-for-InnerStep"),
        ("OuterStep", "token-for-OuterStep"),
    ]


def test_profileblock_still_records_latency_when_hooks_unset(_restore_hooks):
    observability.set_span_hooks(None, None)
    registry = observability.MetricsRegistry.get()
    registry._distributions.clear()

    with observability.ProfileBlock("NoHookStep"):
        pass

    assert "latency::NoHookStep" in registry._distributions


def test_tracing_hooks_are_noop_end_to_end_without_keys(_restore_hooks):
    """utils.tracing registers its own hooks at import time; with no
    Langfuse keys set, running a ProfileBlock through them must not
    raise and must not attempt a client call."""
    tracing.register_hooks()  # idempotent re-registration, mirrors import-time call

    with observability.ProfileBlock("SomeStep"):
        pass  # must not raise
