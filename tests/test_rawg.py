# tests/test_rawg.py
import sys
from pathlib import Path

# Add project root to path — no config files needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from data.rawg import RawgTool


@pytest.fixture
def rawg() -> RawgTool:
    return RawgTool()


def test_search_returns_assassins_creed_shadows(rawg):
    results = rawg.fetch_search("Assassin's Creed Shadows", limit=5)
    assert len(results["results"]) >= 1
    game = results["results"][0]
    assert game["name"] == "Assassin's Creed Shadows"
    assert game["id"] == 981791  # Confirmed live on 2025-11-10


def test_search_cyberpunk_returns_correct_live_id(rawg):
    results = rawg.fetch_search("Cyberpunk 2077", limit=1)
    game = results["results"][0]
    # As of Nov 2025, RAWG reassigned IDs — 41494 is the CURRENT one
    assert game["id"] == 41494
    assert "Cyberpunk 2077" in game["name"]


def test_get_details_has_yasuke_and_samurai(rawg):
    details = rawg.fetch_details(981791)
    assert details["name"] == "Assassin's Creed Shadows"
    assert details["released"] == "2025-03-20"
    assert details["rating"] >= 3.0
    desc = details["description_raw"].lower()
    assert "yasuke" in desc
    assert "shinobi" in desc or "naoe" in desc


def test_fetch_both_returns_consistent_top_game(rawg):
    data = rawg.fetch_both("GTA VI", limit=3)
    search = data["search_results"]
    top = data["top_details"]
    assert len(search) >= 1
    assert top["id"] == search[0]["id"]
    assert "GTA" in top["name"] or "Grand Theft Auto" in top["name"]


def test_fake_game_returns_empty(rawg):
    results = rawg.fetch_search("qwertyuiopgame12345")
    assert len(results["results"]) == 0


def test_empty_query_returns_recent_games(rawg):
    results = rawg.fetch_search("")
    assert len(results["results"]) >= 5


def test_cyberpunk_has_steam_and_playstation(rawg):
    details = rawg.fetch_details(41494)  # Current live ID for Cyberpunk 2077
    stores = [s["store"]["name"] for s in details.get("stores", [])]
    platforms = [p["platform"]["name"] for p in details.get("platforms", [])]
    
    assert any("Steam" in store for store in stores)
    assert any("PlayStation 5" in plat or "PS5" in plat for plat in platforms)
    assert any("Xbox" in plat for plat in platforms)