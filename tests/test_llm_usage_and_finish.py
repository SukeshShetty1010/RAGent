"""
tests/test_llm_usage_and_finish.py

Unit tests for streaming usage accounting and finish-reason tracking.
Fully local — the OpenAI/Groq clients are replaced with fakes, so no
network calls and no API keys are required.

These cover two failures that were silent in production: token counts
that never got recorded (leaving the prompt_tokens/completion_tokens/
cost_usd KPIs null), and answers cut off mid-sentence that the pipeline
still reported as successful.
"""

import pytest

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------
# Fakes shaped like the provider stream chunks
# ------------------------------------------------------------------

class _Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content=None, finish_reason=None):
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Chunk:
    """A streaming chunk. usage/x_groq default to absent, like real ones."""

    def __init__(self, choices=None, usage=None, x_groq=None):
        self.choices = choices or []
        self.usage = usage
        self.x_groq = x_groq


class _XGroq:
    def __init__(self, usage):
        self.usage = usage


# ------------------------------------------------------------------
# _chunk_usage
# ------------------------------------------------------------------

def test_chunk_usage_reads_choiceless_chunk():
    """Groq's shape: a final chunk with no choices carrying usage."""
    from llm.ragent_client import _chunk_usage

    usage = _Usage(10, 20)
    assert _chunk_usage(_Chunk(choices=[], usage=usage)) is usage


def test_chunk_usage_reads_chunk_that_still_has_choices():
    """Gemini's shape, and the regression that nulled every token KPI:
    usage rides on the same final chunk that carries finish_reason, so a
    choice-less-only check never matches."""
    from llm.ragent_client import _chunk_usage

    usage = _Usage(861, 99)
    chunk = _Chunk(choices=[_Choice(content=None, finish_reason="stop")], usage=usage)
    assert _chunk_usage(chunk) is usage


def test_chunk_usage_falls_back_to_x_groq():
    from llm.ragent_client import _chunk_usage

    usage = _Usage(5, 6)
    assert _chunk_usage(_Chunk(choices=[], usage=None, x_groq=_XGroq(usage))) is usage


def test_chunk_usage_absent_returns_none():
    from llm.ragent_client import _chunk_usage

    assert _chunk_usage(_Chunk(choices=[_Choice("hi")])) is None


# ------------------------------------------------------------------
# _record_finish_reason
# ------------------------------------------------------------------

def test_record_finish_reason_ignores_none():
    """Every non-final chunk carries finish_reason=None; recording those
    would swamp the real value."""
    from llm.ragent_client import _record_finish_reason
    from utils.observability import MetricsRegistry

    registry = MetricsRegistry.get()
    registry._categoricals.clear()

    _record_finish_reason(None)
    assert "llm_finish_reason" not in registry.generate_report()["categoricals"]

    _record_finish_reason("length")
    assert registry.generate_report()["categoricals"]["llm_finish_reason"] == {"length": 1}


# ------------------------------------------------------------------
# The streaming client end to end, against a fake Gemini stream
# ------------------------------------------------------------------

def _fake_gemini(chunks):
    """Minimal stand-in for the OpenAI-compat client object."""

    class _Completions:
        def create(self, **_kwargs):
            return iter(chunks)

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


def test_streaming_records_usage_from_final_chunk_with_choices(monkeypatch):
    """The whole bug in one test: Gemini attaches usage to a chunk that
    still has choices. Reading it only from choice-less chunks recorded
    zero tokens for every answer."""
    import llm.ragent_client_streaming as streaming
    from utils.observability import MetricsRegistry

    registry = MetricsRegistry.get()
    registry._distributions.clear()
    registry._categoricals.clear()

    chunks = [
        _Chunk(choices=[_Choice("Hello ")], usage=_Usage(861, 1)),
        _Chunk(choices=[_Choice("world.")], usage=_Usage(861, 3)),
        _Chunk(choices=[_Choice(None, finish_reason="stop")], usage=_Usage(861, 5)),
    ]
    monkeypatch.setattr(streaming, "_get_gemini_client", lambda: _fake_gemini(chunks))

    out = list(streaming.chat_completion_streaming("prompt"))
    assert "".join(out) == "Hello world."

    report = registry.generate_report()
    assert report["distributions"]["llm_prompt_tokens"]["max"] == 861
    assert report["distributions"]["llm_completion_tokens"]["max"] == 5
    assert report["categoricals"]["llm_finish_reason"] == {"stop": 1}


def test_streaming_records_length_finish_reason(monkeypatch):
    """A truncated answer must leave a "length" marker behind, or the
    engine cannot tell it from a complete short answer."""
    import llm.ragent_client_streaming as streaming
    from utils.observability import MetricsRegistry

    registry = MetricsRegistry.get()
    registry._categoricals.clear()

    chunks = [
        _Chunk(choices=[_Choice("A partial senten")], usage=_Usage(100, 4)),
        _Chunk(choices=[_Choice(None, finish_reason="length")], usage=_Usage(100, 4)),
    ]
    monkeypatch.setattr(streaming, "_get_gemini_client", lambda: _fake_gemini(chunks))

    list(streaming.chat_completion_streaming("prompt"))
    assert registry.generate_report()["categoricals"]["llm_finish_reason"] == {"length": 1}


def test_streaming_leaves_no_finish_reason_when_stream_is_cut(monkeypatch):
    """A provider that closes the stream mid-generation (Gemini's free
    tier does this on quota exhaustion) never sends a finish_reason at
    all. The engine treats that absence as truncation, so nothing must
    invent one here."""
    import llm.ragent_client_streaming as streaming
    from utils.observability import MetricsRegistry

    registry = MetricsRegistry.get()
    registry._categoricals.clear()

    chunks = [
        _Chunk(choices=[_Choice("Cut off right ")], usage=_Usage(100, 3)),
    ]
    monkeypatch.setattr(streaming, "_get_gemini_client", lambda: _fake_gemini(chunks))

    list(streaming.chat_completion_streaming("prompt"))
    assert "llm_finish_reason" not in registry.generate_report()["categoricals"]


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def test_answer_budget_exceeds_the_measured_requirement():
    """512 cut a real answer off mid-sentence (finish_reason="length" at
    861 prompt / 692 completion tokens, measured 2026-08-17). The cap
    must stay comfortably above that."""
    from llm.ragent_client import _ANSWER_MAX_TOKENS

    assert _ANSWER_MAX_TOKENS >= 1024


def test_groq_fallback_model_is_not_the_retired_llama():
    """llama-3.1-8b-instant 404s on Groq now, which silently killed the
    entire fallback path."""
    from llm.ragent_client import _GROQ_MODEL

    assert _GROQ_MODEL != "llama-3.1-8b-instant"


def test_groq_fallback_model_is_priced():
    """An unpriced model reports cost_usd 0.0, which is indistinguishable
    from Gemini's genuinely-free tier."""
    from llm.pricing import MODEL_PRICING
    from llm.ragent_client import _GROQ_MODEL

    assert MODEL_PRICING.get(_GROQ_MODEL, (0.0, 0.0)) != (0.0, 0.0)
