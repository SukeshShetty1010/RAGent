"""
tests/test_query_rewrite.py

Hermetic tests for agent/decisions/query_rewrite.py (AUDIT_TASKS T14).
chat_completion_decision is monkeypatched -- no network, no credentials.
"""

import json

import pytest

import agent.decisions.query_rewrite as query_rewrite_mod
from agent.decisions.query_rewrite import (
    HISTORY_MAX_TURNS,
    HISTORY_TURN_CHAR_CAP,
    rewrite_query,
)

pytestmark = pytest.mark.unit


def _boom(*args, **kwargs):
    raise AssertionError("chat_completion_decision should not have been called")


# --------------------------------------------------------------
# End-to-end §14 scenario
# --------------------------------------------------------------

def test_rewrite_resolves_anaphora_using_history(monkeypatch):
    monkeypatch.setattr(
        query_rewrite_mod,
        "chat_completion_decision",
        lambda *a, **kw: json.dumps({
            "rewritten_query": "Far Cry 5 story",
            "reason": "resolved 'its' to Far Cry 5",
        }),
    )

    history = [
        {"role": "user", "content": "Tell me about Far Cry 5"},
        {"role": "assistant", "content": "Far Cry 5 is an open-world FPS set in Hope County."},
    ]

    result = rewrite_query(query="what about its story?", history=history)

    assert result.source == "llm"
    assert "Far Cry 5" in result.rewritten_query
    assert result.original_query == "what about its story?"


# --------------------------------------------------------------
# Deterministic pre-checks -- LLM must never be called
# --------------------------------------------------------------

def test_no_history_is_passthrough(monkeypatch):
    monkeypatch.setattr(query_rewrite_mod, "chat_completion_decision", _boom)

    result = rewrite_query(query="Tell me about Far Cry 5", history=[])

    assert result.source == "skipped_no_history"
    assert result.rewritten_query == "Tell me about Far Cry 5"


def test_none_history_is_passthrough(monkeypatch):
    monkeypatch.setattr(query_rewrite_mod, "chat_completion_decision", _boom)

    result = rewrite_query(query="Tell me about Far Cry 5", history=None)

    assert result.source == "skipped_no_history"


def test_self_contained_query_is_skipped(monkeypatch):
    monkeypatch.setattr(query_rewrite_mod, "chat_completion_decision", _boom)

    history = [{"role": "user", "content": "Tell me about Far Cry 5"}]
    result = rewrite_query(query="What platforms is Far Cry 5 available on?", history=history)

    assert result.source == "skipped_self_contained"
    assert result.rewritten_query == "What platforms is Far Cry 5 available on?"


def test_bare_fragment_triggers_llm(monkeypatch):
    """A short fragment with no anaphora word still needs resolving --
    it isn't self-contained just because it lacks 'it/that/this'."""
    monkeypatch.setattr(
        query_rewrite_mod,
        "chat_completion_decision",
        lambda *a, **kw: json.dumps({"rewritten_query": "Far Cry 5 multiplayer", "reason": "x"}),
    )

    history = [{"role": "user", "content": "Tell me about Far Cry 5"}]
    result = rewrite_query(query="and multiplayer?", history=history)

    assert result.source == "llm"


# --------------------------------------------------------------
# Fail-soft on every failure mode
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "stub",
    [
        pytest.param(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("llm down")), id="exception"),
        pytest.param(lambda *a, **kw: json.dumps({"rewritten_query": ""}), id="empty_output"),
        pytest.param(lambda *a, **kw: "not json", id="malformed_json"),
        pytest.param(
            lambda *a, **kw: json.dumps({"rewritten_query": "x" * 501}),
            id="oversized_output",
        ),
    ],
)
def test_every_failure_mode_falls_back_to_original(monkeypatch, stub):
    monkeypatch.setattr(query_rewrite_mod, "chat_completion_decision", stub)

    history = [{"role": "user", "content": "Tell me about Far Cry 5"}]
    result = rewrite_query(query="what about its story?", history=history)

    assert result.source == "fallback_original"
    assert result.rewritten_query == "what about its story?"


# --------------------------------------------------------------
# History bounding
# --------------------------------------------------------------

def test_history_formatting_truncates_to_max_turns_and_char_cap(monkeypatch):
    captured_prompt = {}

    def _capture(prompt, *a, **kw):
        captured_prompt["prompt"] = prompt
        return json.dumps({"rewritten_query": "resolved query", "reason": "x"})

    monkeypatch.setattr(query_rewrite_mod, "chat_completion_decision", _capture)

    long_turn = "y" * (HISTORY_TURN_CHAR_CAP + 200)
    history = [
        {"role": "user", "content": f"turn-{i}" if i > 0 else long_turn}
        for i in range(HISTORY_MAX_TURNS + 3)
    ]

    rewrite_query(query="what about it?", history=history)

    prompt = captured_prompt["prompt"]
    # Only the last HISTORY_MAX_TURNS turns should appear.
    assert "turn-0" not in prompt
    for i in range(HISTORY_MAX_TURNS + 3 - HISTORY_MAX_TURNS, HISTORY_MAX_TURNS + 3):
        assert f"turn-{i}" in prompt
    # The oversized first turn was truncated out of the window entirely,
    # but even if it had survived, no run of the char exceeding the cap
    # should appear uncapped.
    assert "y" * (HISTORY_TURN_CHAR_CAP + 1) not in prompt


# --------------------------------------------------------------
# max_tokens must not be overridden (AUDIT_TASKS T4 pattern)
# --------------------------------------------------------------

def test_rewrite_does_not_override_max_tokens_default(monkeypatch):
    captured_kwargs = {}

    def _capture(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return json.dumps({"rewritten_query": "resolved query", "reason": "x"})

    monkeypatch.setattr(query_rewrite_mod, "chat_completion_decision", _capture)

    history = [{"role": "user", "content": "Tell me about Far Cry 5"}]
    rewrite_query(query="what about its story?", history=history)

    assert "max_tokens" not in captured_kwargs
