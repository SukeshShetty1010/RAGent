import pytest
import requests  # <-- FIX: Import requests
from unittest.mock import patch, MagicMock
from auth.gamespot_client import GameSpotClient
from data.gamespot_data import GameSpotData, remove_visual_fields, safe_get


# ---------- GameSpotClient Tests ----------

def test_rate_limit(monkeypatch):
    """Ensure rate limiter updates internal timestamp."""
    client = GameSpotClient.__new__(GameSpotClient)
    client.api_key = "test"
    client.user_agent = "pytest"
    client._last_request_time = 0
    client.RATE_LIMIT_SLEEP = 0

    client._rate_limit()
    assert client._last_request_time > 0


@patch("requests.get")
def test_get_success(mock_get):
    """Test _get() method with valid JSON response."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"results": [{"id": 1, "name": "Mock Game"}]}
    mock_get.return_value = mock_resp

    client = GameSpotClient.__new__(GameSpotClient)
    client.api_key = "fake"
    client.user_agent = "pytest"
    client._last_request_time = 0
    result = GameSpotClient._get(client, "games", {"filter": "id:1"})
    assert "results" in result


@patch("requests.get")
def test_get_error(mock_get):
    """Test _get() handles exceptions gracefully."""
    # FIX: Raise the specific exception caught by the _get() method
    mock_get.side_effect = requests.RequestException("Network down")

    client = GameSpotClient.__new__(GameSpotClient)
    client.api_key = "fake"
    client.user_agent = "pytest"
    client._last_request_time = 0
    result = GameSpotClient._get(client, "games", {"filter": "id:1"})
    assert result is None  # This assertion should now be reached


@patch.object(GameSpotClient, "_get", return_value={"results": [{"id": 1}]})
def test_fetch(mock_get):
    """Test fetch() returns parsed list."""
    client = GameSpotClient.__new__(GameSpotClient)
    client.api_key = "fake"
    result = GameSpotClient.fetch(client, "games", "name:GTA")
    assert isinstance(result, list)
    assert result[0]["id"] == 1


@patch.object(GameSpotClient, "fetch", side_effect=[
    [{"id": 1, "name": "page1"}], [],
])
def test_fetch_all_pages(mock_fetch):
    """Test multi-page fetch ends when no results."""
    client = GameSpotClient.__new__(GameSpotClient)
    client.api_key = "fake"
    results = GameSpotClient.fetch_all_pages(client, "games", "name:GTA", max_pages=3)
    assert results[0]["name"] == "page1"


# ---------- Helper Function Tests ----------

def test_remove_visual_fields():
    """Ensure visual keys are removed properly."""
    data = {"image": "url", "name": "GTA"}
    clean = remove_visual_fields(data)
    assert "image" not in clean
    assert "name" in clean


def test_safe_get_nested():
    """Test safe traversal of nested dicts."""
    data = {"a": {"b": {"c": 42}}}
    assert safe_get(data, "a", "b", "c") == 42
    assert safe_get(data, "x") is None


# ---------- GameSpotData Tests ----------

@patch("data.gamespot_data.GameSpotClient")
@patch("data.gamespot_data.RAWGData")
def test_get_game_data_flow(mock_rawg_cls, mock_client_cls):
    """Simulate full flow of GameSpotData.get_game_data()."""
    mock_rawg = MagicMock()
    mock_rawg.search_and_rank_games.return_value = [{"name": "Mock Game"}]
    mock_rawg_cls.return_value = mock_rawg

    mock_client = MagicMock()
    # Simulate client.fetch() finding a game
    mock_client.fetch.side_effect = [
        [{"id": 123, "name": "Mock Game"}]  # search results
    ]
    # Mock fetch_all_pages for each endpoint
    mock_client.fetch_all_pages.side_effect = [
        [{"id": 1, "name": "Mock Game"}],
        [{"id": 2, "platform": {"name": "PC"}}],
        [{"id": 3, "title": "Game Article"}],
        [{"id": 4, "title": "Game Review", "score": 8.5}],
    ]
    mock_client_cls.return_value = mock_client

    gs = GameSpotData()
    result = gs.get_game_data("mock game")

    assert isinstance(result, dict)
    assert "Game Information" in result
    assert "Releases" in result
    assert len(result["Reviews"]) == 1
    assert result["Reviews"][0]["score"] == 8.5


def test_to_hierarchical_structure():
    """Test _to_hierarchical creates expected dictionary."""
    gs = GameSpotData.__new__(GameSpotData)
    mock_data = {
        "game": [{"id": 1, "name": "Mock Game", "release_date": "2021-01-01"}],
        "releases": [{"id": 2, "platform": {"name": "PC"}}],
        "articles": [{"id": 3, "title": "Game Article"}],
        "reviews": [{"id": 4, "title": "Game Review"}],
    }
    structured = GameSpotData._to_hierarchical(gs, mock_data)
    assert structured["Game Information"]["name"] == "Mock Game"
    assert len(structured["Releases"]) == 1
    assert len(structured["Articles"]) == 1
    assert len(structured["Reviews"]) == 1