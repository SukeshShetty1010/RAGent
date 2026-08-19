"""
tests/test_answer_model_attribution.py

Hermetic tests for T6: the model attributed to an answer must be the
model that actually produced the answer, not whichever provider was
merely used somewhere in the request (e.g. a Gemini-served web-search
decision must not make a Groq-served answer look like it came from
Gemini).

Fully local — providers are replaced with fakes shaped like
tests/test_llm_usage_and_finish.py's, so no network calls and no API
keys are required.
"""

import pytest

pytestmark = pytest.mark.unit


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
    def __init__(self, choices=None, usage=None, x_groq=None):
        self.choices = choices or []
        self.usage = usage
        self.x_groq = x_groq


class _Message:
    def __init__(self, content):
        self.content = content


class _NonStreamChoice:
    """Shape returned by create_completion(stream=False): .message, not .delta."""

    def __init__(self, content, finish_reason=None):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Completion:
    def __init__(self, content, usage, finish_reason="stop"):
        self.choices = [_NonStreamChoice(content, finish_reason)]
        self.usage = usage


def _fake_gemini(chunks=None, exc=None):
    """Minimal stand-in for the OpenAI-compat client. Raises `exc` from
    create() when given, instead of returning a chunk stream."""

    class _Completions:
        def create(self, **_kwargs):
            if exc is not None:
                raise exc
            return iter(chunks or [])

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


def _fake_groq(chunks):
    class _Completions:
        def create(self, **_kwargs):
            return iter(chunks)

    class _ChatCompletions:
        completions = _Completions()

    class _Client:
        chat = _ChatCompletions()

    return _Client()


@pytest.fixture(autouse=True)
def _clean_registry():
    from utils.observability import MetricsRegistry

    MetricsRegistry.get().reset()
    yield
    MetricsRegistry.get().reset()


def test_answer_model_unknown_before_any_generation():
    from llm.ragent_client import answer_model

    assert answer_model() == "unknown"


def test_answer_served_by_gemini_is_attributed_to_gemini(monkeypatch):
    import llm.ragent_client_streaming as streaming
    from llm.ragent_client import answer_model
    from llm.gemini_client import GEMINI_MODEL

    chunks = [
        _Chunk(choices=[_Choice("Hello ")]),
        _Chunk(choices=[_Choice("world.", finish_reason="stop")], usage=_Usage(10, 2)),
    ]
    monkeypatch.setattr(streaming, "_get_gemini_client", lambda: _fake_gemini(chunks))

    list(streaming.chat_completion_streaming("prompt"))

    assert answer_model() == GEMINI_MODEL


def test_decision_only_leaves_answer_model_unknown(monkeypatch):
    """A decision call (chat_completion_decision) must not masquerade as
    the model that produced the user-facing answer."""
    from llm import ragent_client
    from llm.gemini_client import GEMINI_MODEL

    completion = _Completion('{"web_search": false}', _Usage(20, 5))
    monkeypatch.setattr(
        ragent_client, "_get_gemini_client", lambda: object()
    )
    monkeypatch.setattr(
        ragent_client, "create_completion", lambda *a, **k: completion
    )

    ragent_client.chat_completion_decision("decide something")

    report = ragent_client.MetricsRegistry.get().generate_report()
    assert report["categoricals"]["decision_model"] == {GEMINI_MODEL: 1}
    assert "answer_model" not in report["categoricals"]
    assert ragent_client.answer_model() == "unknown"


def test_decision_on_gemini_then_answer_falls_back_to_groq(monkeypatch):
    """The exact §6 scenario: decide_web_search() succeeds on Gemini, but
    the answer itself fails over to Groq mid-request. answer_model() must
    report Groq, not Gemini — this is the case that was silently wrong
    before the fix."""
    from llm import ragent_client
    import llm.ragent_client_streaming as streaming
    from llm.ragent_client import _GROQ_MODEL

    decision_completion = _Completion('{"web_search": true}', _Usage(15, 4))
    monkeypatch.setattr(ragent_client, "_get_gemini_client", lambda: object())
    monkeypatch.setattr(
        ragent_client, "create_completion", lambda *a, **k: decision_completion
    )
    ragent_client.chat_completion_decision("should I search the web?")

    monkeypatch.setattr(
        streaming, "_get_gemini_client", lambda: _fake_gemini(exc=RuntimeError("gemini down"))
    )
    groq_chunks = [
        _Chunk(choices=[_Choice("Fallback answer.", finish_reason="stop")], usage=_Usage(30, 6)),
    ]
    monkeypatch.setattr(streaming, "_get_groq_client", lambda: _fake_groq(groq_chunks))

    out = list(streaming.chat_completion_streaming("answer the question"))

    assert "".join(out) == "Fallback answer."
    assert ragent_client.answer_model() == _GROQ_MODEL


def test_mid_stream_gemini_failure_after_tokens_still_attributes_gemini(monkeypatch):
    """Once a chunk has reached the SSE stream, there is no fallback (see
    ragent_client_streaming.py's any_yielded branch) — but the text the
    user received did come from Gemini, so the trace must say so instead
    of leaving answer_model at "unknown"."""
    import llm.ragent_client_streaming as streaming
    from llm.ragent_client import answer_model
    from llm.gemini_client import GEMINI_MODEL

    def _raising_stream():
        yield _Chunk(choices=[_Choice("Partial ")])
        raise RuntimeError("gemini died mid-stream")

    class _Completions:
        def create(self, **_kwargs):
            return _raising_stream()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(streaming, "_get_gemini_client", lambda: _Client())

    out = list(streaming.chat_completion_streaming("prompt"))

    assert "".join(out) == "Partial "
    assert answer_model() == GEMINI_MODEL
