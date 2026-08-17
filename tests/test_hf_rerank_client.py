"""
tests/test_hf_rerank_client.py

Hermetic tests for llm/hf_rerank_client.py. requests is monkeypatched at
the client's import site, so no network access and no live Space.

The Space returns scores already in input order (unlike Voyage, which
sorts by descending relevance), so the contract worth pinning here is
the length check: a truncated response must raise rather than be zipped
silently onto the wrong candidates.
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
def _space_config(monkeypatch):
    # The client calls load_dotenv() inside its getters, which would
    # re-populate real credentials from the developer's .env after the
    # deletes below and leak them into assertions. Neutralize it first —
    # these tests must never read or echo a real token.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)

    monkeypatch.setenv("HF_RERANK_URL", "https://example-space.hf.space")
    monkeypatch.delenv("HF_RERANK_SECRET", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_scores_returned_in_input_order(monkeypatch):
    from llm import hf_rerank_client

    payload = {"scores": [7.2, -4.0, 1.5], "model": "x", "elapsed_ms": 12}
    monkeypatch.setattr(
        hf_rerank_client.requests, "post", lambda *a, **kw: _FakeResponse(payload)
    )

    assert hf_rerank_client.rerank("q", ["a", "b", "c"]) == [7.2, -4.0, 1.5]


def test_length_mismatch_raises(monkeypatch):
    """A short response must not be zipped onto the candidate list — that
    would attach scores to the wrong chunks and silently corrupt ranking."""
    from llm import hf_rerank_client

    payload = {"scores": [7.2], "model": "x", "elapsed_ms": 12}
    monkeypatch.setattr(
        hf_rerank_client.requests, "post", lambda *a, **kw: _FakeResponse(payload)
    )

    with pytest.raises(ValueError):
        hf_rerank_client.rerank("q", ["a", "b", "c"])


def test_empty_documents_short_circuits(monkeypatch):
    from llm import hf_rerank_client

    def _boom(*a, **kw):
        raise AssertionError("should not call the Space for zero documents")

    monkeypatch.setattr(hf_rerank_client.requests, "post", _boom)
    assert hf_rerank_client.rerank("q", []) == []


def test_missing_url_raises(monkeypatch):
    from llm import hf_rerank_client

    monkeypatch.delenv("HF_RERANK_URL", raising=False)

    with pytest.raises(ValueError):
        hf_rerank_client.rerank("q", ["a"])


def test_url_is_normalized(monkeypatch):
    """Quoted/trailing-slash values must not produce a double-slash path —
    `docker run --env-file` passes literal quotes straight through."""
    from llm import hf_rerank_client

    monkeypatch.setenv("HF_RERANK_URL", '"https://example-space.hf.space/" ')
    assert hf_rerank_client._get_base_url() == "https://example-space.hf.space"


def test_auth_headers_only_sent_when_configured(monkeypatch):
    from llm import hf_rerank_client

    assert "X-Rerank-Key" not in hf_rerank_client._headers()
    assert "Authorization" not in hf_rerank_client._headers()

    monkeypatch.setenv("HF_RERANK_SECRET", "s3cret")
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    headers = hf_rerank_client._headers()
    assert headers["X-Rerank-Key"] == "s3cret"
    assert headers["Authorization"] == "Bearer hf_abc"


def test_server_error_retries_then_raises(monkeypatch):
    """A 5xx is how a slept Space reports that it is waking up, so it earns
    exactly one retry."""
    from llm import hf_rerank_client

    calls = []

    def _fake_post(*a, **kw):
        calls.append(1)
        return _FakeResponse({}, status_code=503)

    monkeypatch.setattr(hf_rerank_client.requests, "post", _fake_post)
    monkeypatch.setattr(hf_rerank_client.time, "sleep", lambda _s: None)

    with pytest.raises(requests.exceptions.HTTPError):
        hf_rerank_client.rerank("q", ["a"])
    assert len(calls) == hf_rerank_client._MAX_RETRIES


def test_client_error_raises_without_retry(monkeypatch):
    from llm import hf_rerank_client

    calls = []

    def _fake_post(*a, **kw):
        calls.append(1)
        return _FakeResponse({}, status_code=401)

    monkeypatch.setattr(hf_rerank_client.requests, "post", _fake_post)

    with pytest.raises(requests.exceptions.HTTPError):
        hf_rerank_client.rerank("q", ["a"])
    assert len(calls) == 1


def test_timeout_propagates(monkeypatch):
    """Must raise, not hang: api/main.py's SSE generator blocks on an
    unbounded queue.get(), so a wedged retrieval thread never completes."""
    from llm import hf_rerank_client

    def _fake_post(*a, **kw):
        raise requests.exceptions.ReadTimeout("timed out")

    monkeypatch.setattr(hf_rerank_client.requests, "post", _fake_post)
    monkeypatch.setattr(hf_rerank_client.time, "sleep", lambda _s: None)

    with pytest.raises(requests.exceptions.RequestException):
        hf_rerank_client.rerank("q", ["a"])


def test_explicit_timeout_is_passed(monkeypatch):
    from llm import hf_rerank_client

    captured = {}

    def _fake_post(url, **kw):
        captured.update(kw)
        return _FakeResponse({"scores": [1.0], "model": "x", "elapsed_ms": 1})

    monkeypatch.setattr(hf_rerank_client.requests, "post", _fake_post)
    hf_rerank_client.rerank("q", ["a"])

    assert captured["timeout"] == hf_rerank_client._TIMEOUT


def test_usage_is_recorded(monkeypatch):
    from llm import hf_rerank_client
    from utils.usage_counter import UsageCounter

    monkeypatch.setattr(
        hf_rerank_client.requests,
        "post",
        lambda *a, **kw: _FakeResponse({"scores": [1.0], "model": "x", "elapsed_ms": 1}),
    )
    counter = UsageCounter()
    monkeypatch.setattr(UsageCounter, "get", classmethod(lambda cls: counter))

    hf_rerank_client.rerank("q", ["a"])

    key = next(iter(counter._counts))
    assert key[0] == "hfspace" and key[1] == "rerank"
    assert counter._counts[key]["requests"] == 1
