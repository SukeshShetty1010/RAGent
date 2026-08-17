"""
tests/test_voyage_client.py

Hermetic tests for llm/voyage_client.py. requests.post is monkeypatched
at the client's import site, so no network access and no VOYAGE_API_KEY
is required.

The score-ordering test is the important one: Voyage returns `data`
sorted by descending relevance, not in input order, so reading the
scores positionally would silently attach every score to the wrong
chunk — a bug that produces plausible-looking numbers and corrupts
retrieval ranking rather than failing loudly.
"""

import pytest
import requests

pytestmark = pytest.mark.unit


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")


def test_scores_are_remapped_to_input_order(monkeypatch):
    from llm import voyage_client

    # Response deliberately out of input order (Voyage sorts by score desc).
    payload = {
        "data": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.1},
        ],
        "usage": {"total_tokens": 42},
    }
    monkeypatch.setattr(
        voyage_client.requests, "post", lambda *a, **kw: _FakeResponse(payload)
    )

    scores = voyage_client.rerank("q", ["doc0", "doc1", "doc2"])
    assert scores == [0.5, 0.1, 0.9]


def test_empty_documents_short_circuits(monkeypatch):
    from llm import voyage_client

    def _boom(*a, **kw):
        raise AssertionError("should not call the API for zero documents")

    monkeypatch.setattr(voyage_client.requests, "post", _boom)
    assert voyage_client.rerank("q", []) == []


def test_usage_is_recorded(monkeypatch):
    from llm import voyage_client
    from utils.usage_counter import UsageCounter

    payload = {
        "data": [{"index": 0, "relevance_score": 0.7}],
        "usage": {"total_tokens": 123},
    }
    monkeypatch.setattr(
        voyage_client.requests, "post", lambda *a, **kw: _FakeResponse(payload)
    )

    counter = UsageCounter()
    monkeypatch.setattr(UsageCounter, "get", classmethod(lambda cls: counter))

    voyage_client.rerank("q", ["doc0"])

    key = next(iter(counter._counts))
    assert key[0] == "voyage" and key[1] == "rerank"
    assert counter._counts[key]["requests"] == 1
    assert counter._counts[key]["prompt_tokens"] == 123


def test_client_error_raises_without_retry(monkeypatch):
    """A 401/400 is a config error — retrying it only wastes the caller's
    latency budget, which on the SSE path is the user's wall time."""
    from llm import voyage_client

    calls = []

    def _fake_post(*a, **kw):
        calls.append(1)
        return _FakeResponse({}, status_code=401)

    monkeypatch.setattr(voyage_client.requests, "post", _fake_post)

    with pytest.raises(requests.exceptions.HTTPError):
        voyage_client.rerank("q", ["doc0"])
    assert len(calls) == 1


def test_server_error_retries_then_raises(monkeypatch):
    from llm import voyage_client

    calls = []

    def _fake_post(*a, **kw):
        calls.append(1)
        return _FakeResponse({}, status_code=503)

    monkeypatch.setattr(voyage_client.requests, "post", _fake_post)
    monkeypatch.setattr(voyage_client.time, "sleep", lambda _s: None)

    with pytest.raises(requests.exceptions.HTTPError):
        voyage_client.rerank("q", ["doc0"])
    assert len(calls) == voyage_client._MAX_RETRIES


def test_timeout_retries_then_raises(monkeypatch):
    """The timeout must propagate rather than hang: api/main.py's SSE
    generator blocks on an unbounded queue.get(), so a wedged retrieval
    thread never completes the stream."""
    from llm import voyage_client

    calls = []

    def _fake_post(*a, **kw):
        calls.append(1)
        raise requests.exceptions.ConnectTimeout("timed out")

    monkeypatch.setattr(voyage_client.requests, "post", _fake_post)
    monkeypatch.setattr(voyage_client.time, "sleep", lambda _s: None)

    with pytest.raises(requests.exceptions.RequestException):
        voyage_client.rerank("q", ["doc0"])
    assert len(calls) == voyage_client._MAX_RETRIES


def test_explicit_timeout_is_passed(monkeypatch):
    from llm import voyage_client

    captured = {}

    def _fake_post(url, **kw):
        captured.update(kw)
        return _FakeResponse(
            {"data": [{"index": 0, "relevance_score": 0.3}], "usage": {}}
        )

    monkeypatch.setattr(voyage_client.requests, "post", _fake_post)
    voyage_client.rerank("q", ["doc0"])

    assert captured["timeout"] == voyage_client._TIMEOUT


def test_missing_api_key_raises(monkeypatch):
    from llm import voyage_client

    # load_dotenv is imported inside the key getter; neutralize it so a
    # real local .env cannot leak a key into this test.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    with pytest.raises(ValueError):
        voyage_client.rerank("q", ["doc0"])


def test_quoted_api_key_is_stripped(monkeypatch):
    """`docker run --env-file` passes literal quotes through; an unstripped
    quote 401s every request (hit live once with GEMINI_API_KEY)."""
    from llm import voyage_client

    monkeypatch.setenv("VOYAGE_API_KEY", '"quoted-key"')
    assert voyage_client._get_voyage_api_key() == "quoted-key"
