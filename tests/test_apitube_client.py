import os
import pytest
from api.apitube_client import APITubeClient

def test_init_no_key(monkeypatch):
    """Test that client raises if neither arg nor env key is provided."""
    monkeypatch.delenv("APITUBE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        APITubeClient(api_key=None)

def test_init_with_env_key(monkeypatch):
    monkeypatch.setenv("APITUBE_API_KEY", "dummykey_env")
    client = APITubeClient(api_key=None)
    assert client.api_key == "dummykey_env"

def test_init_with_arg_key(monkeypatch):
    monkeypatch.setenv("APITUBE_API_KEY", "dummykey_env")
    client = APITubeClient(api_key="explicit_key")
    assert client.api_key == "explicit_key"

def test_get_news_basic(monkeypatch):
    """Test basic news fetch (live)."""
    if "APITUBE_API_KEY" not in os.environ:
        pytest.skip("APITUBE_API_KEY not set in environment — skipping live API test")

    client = APITubeClient()
    resp = client.get_news(q="technology", limit=3)

    # It should be a dict, and should contain "results"
    assert isinstance(resp, dict), f"Expected dict, got {type(resp)}"
    assert "results" in resp, f"No 'results' in response keys: {list(resp.keys())}"

    results = resp["results"]
    assert isinstance(results, list), f"'results' should be list, got {type(results)}"
    if results:
        first = results[0]
        assert isinstance(first, dict)
        # check some expected fields in article object
        # e.g. title, published_at, source, url (may vary)
        assert "title" in first or "published_at" in first or "source" in first

def test_get_top_headlines(monkeypatch):
    if "APITUBE_API_KEY" not in os.environ:
        pytest.skip("APITUBE_API_KEY not set — skipping live API test")

    client = APITubeClient()
    resp = client.get_top_headlines(country="us", limit=5)
    assert isinstance(resp, dict)
    assert "results" in resp, f"No 'results' in response keys: {list(resp.keys())}"
    results = resp["results"]
    assert isinstance(results, list)
    if results:
        first = results[0]
        assert isinstance(first, dict)
        assert "title" in first or "published_at" in first or "source" in first