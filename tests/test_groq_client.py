import pytest
import os
from llm.ragent_client import chat_completion_remote

@pytest.mark.live
def test_groq_client_basic_response():
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
        
    response = chat_completion_remote("Say 'hello test'", max_tokens=10)
    assert len(response) > 0
    assert "hello" in response.lower()
