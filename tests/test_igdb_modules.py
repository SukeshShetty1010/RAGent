import pytest
from unittest.mock import patch, MagicMock
from auth.igdb_client import IGDBClient
from data.igdb_data import IGDBData, _epoch_to_iso


# ---------- IGDBClient Tests ----------

def test_token_refresh(monkeypatch):
    """Ensure that IGDBClient fetches a new token when expired."""
    client = IGDBClient.__new__(IGDBClient)
    client.client_id = "fake_id"
    client.client_secret = "fake_secret"
    client._token = None
    client._token_expiry = 0

    fake_token = "abc123"

    def mock_post(url, params=None, timeout=10):
        class FakeResponse:
            def raise_for_status(self): pass
            def json(self): return {"access_token": fake_token, "expires_in": 3600}
        return FakeResponse()

    with patch("requests.post", mock_post):
        token = IGDBClient._fetch_new_token(client)
        assert token == fake_token
        assert client._token == fake_token


def test_post_method(monkeypatch):
    """Ensure IGDBClient.post handles successful API responses."""
    client = IGDBClient.__new__(IGDBClient)
    client.client_id = "fake_id"
    client.client_secret = "fake_secret"
    client._token = "fake_token"
    client._token_expiry = 9999999999
    client._last_request_time = 0

    def mock_post(url, headers=None, data=None, timeout=15):
        class FakeResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return [{"id": 1, "name": "Mock Game"}]
        return FakeResponse()

    with patch("requests.post", mock_post):
        res = IGDBClient.post(client, "games", "fields name;")
        assert isinstance(res, list)
        assert res[0]["name"] == "Mock Game"


# ---------- Utility Function Tests ----------

def test_epoch_to_iso_valid():
    """Test correct conversion from epoch timestamp to ISO format."""
    ts = 1609459200  # 2021-01-01 UTC
    iso = _epoch_to_iso(ts)
    assert iso.startswith("2021-01-01T00:00:00")

def test_epoch_to_iso_invalid():
    """Test graceful handling of invalid timestamps."""
    assert _epoch_to_iso(None) is None
    assert _epoch_to_iso("invalid") is None


# ---------- IGDBData Tests ----------

@patch("data.igdb_data.RAWGData")
@patch("data.igdb_data.IGDBClient")
def test_get_game_by_name(mock_client_cls, mock_rawg_cls):
    """Simulate RAWG + IGDB flow and ensure normalized data returned."""
    # Mock RAWG
    mock_rawg = MagicMock()
    mock_rawg.search_and_rank_games.return_value = [{"name": "Grand Theft Auto V"}]
    mock_rawg_cls.return_value = mock_rawg

    # Mock IGDBClient
    mock_client = MagicMock()
    mock_client.post.side_effect = [
        [{"id": 1, "name": "Grand Theft Auto V", "genres": [10], "first_release_date": 1609459200}],
        [{"id": 10, "name": "Action"}],  # genre lookup
    ]
    mock_client_cls.return_value = mock_client

    data = IGDBData()
    result = data.get_game_by_name("GTA 5")

    assert isinstance(result, dict)
    assert result["name"] == "Grand Theft Auto V"
    assert result["genres"] == ["Action"]
    assert result["release_date"].startswith("2021-01-01T")


@patch("data.igdb_data.IGDBClient.post")
def test_fetch_lookup(mock_post):
    """Ensure lookup method returns correct mapping."""
    mock_post.return_value = [{"id": 1, "name": "Mock Genre"}]
    client = IGDBData()
    client.client = MagicMock()
    client.client.post = mock_post

    res = client._fetch_lookup("genres", [1])
    assert 1 in res
    assert res[1]["name"] == "Mock Genre"
