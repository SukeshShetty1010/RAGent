"""
tests/test_cloudflare_rerank_client.py

Hermetic tests for llm/cloudflare_rerank_client.py. requests is
monkeypatched at the client's import site — no network, no credentials.

The remap tests carry the weight here. Cloudflare documents that the
response refers back to each context's index but does not document
whether results come back sorted, so the client treats the id field as
authoritative. Reading a sorted response positionally would attach every
score to the wrong chunk and still look like plausible numbers.
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


def _ok(entries):
    return {"result": {"response": entries}, "success": True, "errors": [], "messages": []}


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token123")


def test_scores_remapped_by_id_when_response_is_sorted(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    payload = _ok([
        {"id": 2, "score": 8.5},
        {"id": 0, "score": 1.0},
        {"id": 1, "score": -4.2},
    ])
    monkeypatch.setattr(cf.requests, "post", lambda *a, **kw: _FakeResponse(payload))

    assert cf.rerank("q", ["a", "b", "c"]) == [1.0, -4.2, 8.5]


def test_index_field_is_accepted_as_an_alias(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    payload = _ok([{"index": 1, "score": 3.0}, {"index": 0, "score": 2.0}])
    monkeypatch.setattr(cf.requests, "post", lambda *a, **kw: _FakeResponse(payload))

    assert cf.rerank("q", ["a", "b"]) == [2.0, 3.0]


def test_partial_response_raises(monkeypatch):
    """Every candidate needs a score — a short response must not be
    silently padded with zeros, which on a logit scale reads as
    'moderately relevant' rather than 'unknown'."""
    from llm import cloudflare_rerank_client as cf

    payload = _ok([{"id": 0, "score": 5.0}])
    monkeypatch.setattr(cf.requests, "post", lambda *a, **kw: _FakeResponse(payload))

    with pytest.raises(ValueError):
        cf.rerank("q", ["a", "b", "c"])


def test_out_of_range_index_raises(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    payload = _ok([{"id": 0, "score": 5.0}, {"id": 9, "score": 1.0}])
    monkeypatch.setattr(cf.requests, "post", lambda *a, **kw: _FakeResponse(payload))

    with pytest.raises(ValueError):
        cf.rerank("q", ["a", "b"])


def test_unexpected_shape_raises(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    monkeypatch.setattr(
        cf.requests, "post", lambda *a, **kw: _FakeResponse({"result": {}, "success": True})
    )
    with pytest.raises(ValueError):
        cf.rerank("q", ["a"])


def test_success_false_raises(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    payload = {"result": None, "success": False, "errors": [{"code": 7003, "message": "no route"}]}
    monkeypatch.setattr(cf.requests, "post", lambda *a, **kw: _FakeResponse(payload))

    with pytest.raises(ValueError):
        cf.rerank("q", ["a"])


def test_top_k_is_not_sent(monkeypatch):
    """top_k would truncate the results and leave most candidates
    unscored — the caller needs one score per document."""
    from llm import cloudflare_rerank_client as cf

    captured = {}

    def _fake_post(url, **kw):
        captured.update(kw)
        return _FakeResponse(_ok([{"id": 0, "score": 1.0}]))

    monkeypatch.setattr(cf.requests, "post", _fake_post)
    cf.rerank("q", ["a"])

    assert "top_k" not in captured["json"]
    assert captured["json"]["contexts"] == [{"text": "a"}]
    assert captured["timeout"] == cf._TIMEOUT


def test_endpoint_includes_account_and_model(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    url = cf._endpoint()
    assert "accounts/acct123/ai/run/" in url
    assert cf.CLOUDFLARE_RERANK_MODEL in url


def test_missing_credentials_raise(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError):
        cf.rerank("q", ["a"])


def test_quoted_credentials_are_stripped(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", '"acct999"')
    assert "accounts/acct999/" in cf._endpoint()


def test_empty_documents_short_circuits(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    def _boom(*a, **kw):
        raise AssertionError("should not call the API for zero documents")

    monkeypatch.setattr(cf.requests, "post", _boom)
    assert cf.rerank("q", []) == []


def test_rate_limit_retries_then_raises(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    calls = []

    def _fake_post(*a, **kw):
        calls.append(1)
        return _FakeResponse({}, status_code=429)

    monkeypatch.setattr(cf.requests, "post", _fake_post)
    monkeypatch.setattr(cf.time, "sleep", lambda _s: None)

    with pytest.raises(requests.exceptions.HTTPError):
        cf.rerank("q", ["a"])
    assert len(calls) == cf._MAX_RETRIES


def test_client_error_raises_without_retry(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    calls = []

    def _fake_post(*a, **kw):
        calls.append(1)
        return _FakeResponse({}, status_code=403)

    monkeypatch.setattr(cf.requests, "post", _fake_post)

    with pytest.raises(requests.exceptions.HTTPError):
        cf.rerank("q", ["a"])
    assert len(calls) == 1


def test_timeout_propagates(monkeypatch):
    from llm import cloudflare_rerank_client as cf

    monkeypatch.setattr(
        cf.requests, "post",
        lambda *a, **kw: (_ for _ in ()).throw(requests.exceptions.ReadTimeout("timed out")),
    )
    monkeypatch.setattr(cf.time, "sleep", lambda _s: None)

    with pytest.raises(requests.exceptions.RequestException):
        cf.rerank("q", ["a"])


def test_usage_is_recorded(monkeypatch):
    from llm import cloudflare_rerank_client as cf
    from utils.usage_counter import UsageCounter

    monkeypatch.setattr(
        cf.requests, "post", lambda *a, **kw: _FakeResponse(_ok([{"id": 0, "score": 1.0}]))
    )
    counter = UsageCounter()
    monkeypatch.setattr(UsageCounter, "get", classmethod(lambda cls: counter))

    cf.rerank("q", ["a"])

    key = next(iter(counter._counts))
    assert key[0] == "cloudflare" and key[1] == "rerank"
    assert counter._counts[key]["requests"] == 1
