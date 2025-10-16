import os
import pytest
import requests
from api.igdb_client import igdb_request
from api.auth import get_igdb_token

def test_get_igdb_token_env_vars():
    if "TWITCH_CLIENT_ID" not in os.environ or "TWITCH_CLIENT_SECRET" not in os.environ:
        pytest.skip("Twitch credentials missing — skipping live token test")

    token = get_igdb_token()
    assert isinstance(token, str)
    assert len(token) > 0

def test_igdb_request_basic():
    if "TWITCH_CLIENT_ID" not in os.environ or "TWITCH_CLIENT_SECRET" not in os.environ:
        pytest.skip("Twitch credentials missing — skip live IGDB test")

    resp = igdb_request("games", "fields id,name; limit 2;")
    assert isinstance(resp, list)
    for item in resp:
        assert "id" in item
        assert "name" in item

def test_igdb_request_auth_failure(monkeypatch):
    """Mock requests.post to return 401 so igdb_request raises HTTPError."""
    # Force token to some value (doesn't matter)
    monkeypatch.setattr("api.auth.get_igdb_token", lambda: "invalid_token")

    class DummyResp:
        def __init__(self):
            self.status_code = 401
        def raise_for_status(self):
            raise requests.HTTPError("401 Unauthorized")
        def json(self):
            return {"error": "unauthorized"}

    def fake_post(url, data=None, headers=None):
        return DummyResp()

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(requests.HTTPError):
        igdb_request("games", "fields id,name; limit 1;")

def test_igdb_request_mocked(monkeypatch):
    """Mock both token fetch and HTTP call to return fake payload."""
    fake_payload = [{"id": 111, "name": "MockGame"}]

    # Patch get_igdb_token to always return dummy token
    monkeypatch.setattr("api.auth.get_igdb_token", lambda: "dummy_token")

    class DummyRespOK:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_success_post(url, data=None, headers=None):
        # optionally assert headers contain bearer token
        assert headers is not None
        assert headers.get("Authorization", "").startswith("Bearer ")
        return DummyRespOK(fake_payload)

    monkeypatch.setattr("requests.post", fake_success_post)

    result = igdb_request("games", "fields id,name; limit 1;")
    assert result == fake_payload

if __name__ == "__main__":
    print("Running IGDB tests...")
    test_get_igdb_token_env_vars()
    print("token test passed")
    test_igdb_request_basic()
    print("basic request test passed")
    test_igdb_request_auth_failure()
    print("auth failure test passed")
    test_igdb_request_mocked()
    print("mocked request test passed")
    print("All tests passed.")