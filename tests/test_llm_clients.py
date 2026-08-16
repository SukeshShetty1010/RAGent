import pytest
import os
from llm.ragent_client import chat_completion_remote


@pytest.mark.live
def test_gemini_primary_basic_response():
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    response = chat_completion_remote("Say 'hello test'", max_tokens=10)
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

    response = chat_completion_remote("Say 'hello test'", max_tokens=10)
    assert len(response) > 0
    assert "hello" in response.lower()
