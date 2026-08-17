import pytest
import os
from llm.ragent_client import chat_completion_remote

# max_tokens=10 used to be enough here, but the Groq fallback is now a
# reasoning model that bills hidden reasoning tokens against max_tokens
# and cannot be told to skip them (Groq accepts only low/medium/high for
# reasoning_effort). At 10 the budget is spent before any content is
# emitted and the call returns "" with finish_reason="length" — a test
# failure that says nothing about whether the client works. 256 is still
# far below any real generation budget.
_SMOKE_MAX_TOKENS = 256


@pytest.mark.live
def test_gemini_primary_basic_response():
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    response = chat_completion_remote("Say 'hello test'", max_tokens=_SMOKE_MAX_TOKENS)
    assert len(response) > 0
    assert "hello" in response.lower()


@pytest.mark.live
def test_groq_fallback_basic_response(monkeypatch):
    """Force the Gemini branch to fail so this exercises the real Groq
    fallback path, not whichever provider happens to answer first."""
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")

    import llm.ragent_client as ragent_client

    def _broken_gemini_client():
        raise ValueError("forced failure for fallback test")

    monkeypatch.setattr(ragent_client, "_get_gemini_client", _broken_gemini_client)

    response = chat_completion_remote("Say 'hello test'", max_tokens=_SMOKE_MAX_TOKENS)
    assert len(response) > 0
    assert "hello" in response.lower()
