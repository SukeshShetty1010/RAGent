# tests/test_rawg_data.py
from unittest.mock import MagicMock

import pytest
from data.rawg_data import RAWGData


# tests/test_rawg_data.py
def _mock_client(search=None, detail=None, rels=None):
    c = MagicMock()
    c.search_games.return_value = search or []
    c.get_game_details.return_value = detail or {}

    rels = rels or {}
    for name in ("additions", "game-series", "achievements", "stores"):
        # FIX: "achievements" and "stores" do not have the 'get_game_' prefix
        if name in ("stores", "achievements"):
            method = f"get_{name}"
        else:
            method = f"get_game_{name.replace('-', '_')}"
        setattr(c, method, lambda *a, n=name: rels.get(n, {"results": []}))
    return c


def test_search_and_rank_games_mocked():
    client = _mock_client(search=[{"id": 1, "name": "Portal", "slug": "portal"}])
    data = RAWGData(client=client)
    out = data.search_and_rank_games("portal", top_k=1)
    assert out[0]["name"] == "Portal"


def test_get_full_game_profile_mocked():
    client = _mock_client(
        detail={
            "id": 1,
            "name": "Portal",
            "description_raw": "Think with portals.",
            "genres": [{"name": "Puzzle"}],
            "tags": [{"name": "First-Person"}] * 25,
            "platforms": [
                {"platform": {"name": "PC"}, "requirements": {"minimum": "2GB RAM"}},
                {"platform": {"name": "Xbox 360"}},
            ],
            "esrb_rating": {"name": "E10+"},
        },
        rels={
            "additions": {"results": [{"name": "Portal 2"}]},
            "game-series": {"results": [{"name": "Half-Life"}]},
            "achievements": {"results": [{"name": "You Monster", "description": "Complete the game."}]},
            "stores": {"results": [{"store": {"name": "Steam"}, "url": "https://store.steampowered.com"}]},
        },
    )

    data = RAWGData(client=client)
    p = data.get_full_game_profile(1)

    assert p["name"] == "Portal"
    assert p["description"] == "Think with portals."
    assert p["genres"] == ["Puzzle"]
    assert len(p["tags"]) == 20
    assert p["system_requirements"] == {"PC": {"minimum": "2GB RAM", "recommended": None}}
    assert p["additions"] == ["Portal 2"]
    assert p["achievements_sample"][0]["name"] == "You Monster"
    assert p["stores"][0]["name"] == "Steam"


def test_get_game_by_name_mocked():
    client = _mock_client(
        search=[{"id": 99, "name": "Mock Game"}],
        detail={"id": 99, "name": "Mock Game", "description_raw": "A test game."}
    )
    data = RAWGData(client=client)
    game = data.get_game_by_name("Mock Game")
    assert game["id"] == 99


def test_search_and_get_full_live(rawg_client):
    data = RAWGData(client=rawg_client)
    candidates = data.search_and_rank_games("portal", top_k=1)
    assert candidates
    game_id = candidates[0]["id"]
    full = data.get_full_game_profile(game_id)
    assert full is not None
    assert full["id"] == game_id
    assert "portal" in full["name"].lower()
    assert isinstance(full["description"], str) and full["description"]
    if full["stores"]:
        assert any(s.get("name") for s in full["stores"] if s.get("name"))


def test_get_game_by_name_live(rawg_client):
    data = RAWGData(client=rawg_client)
    game = data.get_game_by_name("Portal")
    assert game is not None
    assert "portal" in game["name"].lower()
    assert isinstance(game["description"], str) and game["description"]