# tests/test_rawg_client.py
import time

import pytest
import responses
from auth.rawg_client import RAWGClient


# ----------------------------------------------------------------------
# UNIT TESTS – fully mocked
# ----------------------------------------------------------------------
@responses.activate
def test_search_games_mocked():
    body = {"count": 1, "results": [{"id": 1, "name": "Portal", "slug": "portal"}]}
    responses.add(responses.GET, "https://api.rawg.io/api/games", json=body, status=200)

    client = RAWGClient(api_key="test")
    out = client.search_games("portal", page_size=5)

    assert len(out) == 1
    assert out[0]["name"] == "Portal"


@responses.activate
def test_get_game_details_mocked():
    body = {"id": 42, "name": "Portal", "description_raw": "Think with portals."}
    responses.add(responses.GET, "https://api.rawg.io/api/games/42", json=body, status=200)

    client = RAWGClient(api_key="test")
    data = client.get_game_details(42)
    assert data["name"] == "Portal"


@responses.activate
def test_rate_limit_enforced():
    client = RAWGClient(api_key="test")
    client.RATE_LIMIT_SLEEP = 0.05
    responses.add(responses.GET, "https://api.rawg.io/api/games", json={}, status=200)

    start = time.time()
    client._get("/games")
    client._get("/games")
    assert time.time() - start >= 0.04


@responses.activate
def test_429_retry():
    responses.add(responses.GET, "https://api.rawg.io/api/games", status=429)
    responses.add(responses.GET, "https://api.rawg.io/api/games", json={}, status=200)

    client = RAWGClient(api_key="test")
    assert client._get("/games") == {}
    assert len(responses.calls) == 2


@responses.activate
def test_non_200_returns_none():
    responses.add(responses.GET, "https://api.rawg.io/api/games", status=500, body="boom")
    client = RAWGClient(api_key="test")
    assert client._get("/games") is None


# ----------------------------------------------------------------------
# INTEGRATION TESTS – real API (search-then-fetch pattern)
# ----------------------------------------------------------------------
def test_search_live(rawg_client):
    results = rawg_client.search_games("portal", page_size=3)
    assert isinstance(results, list)
    assert any("portal" in r.get("name", "").lower() for r in results)


def test_get_game_details_live(rawg_client):
    # 1. search for a known title
    hits = rawg_client.search_games("portal", page_size=1)
    assert hits, "search returned nothing"
    game_id = hits[0]["id"]

    # 2. fetch details
    game = rawg_client.get_game_details(game_id)
    assert game is not None
    assert game["id"] == game_id
    assert "portal" in game.get("name", "").lower()
    # description can be in either field
    assert game.get("description_raw") or game.get("description")