# tests/test_igdb.py
import os
import json
import pytest
from datetime import datetime
from pathlib import Path
from datetime import UTC

# FORCE LOAD .env at the very top!
from dotenv import load_dotenv
load_dotenv()  # This line fixes everything

# Now add project root
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from data.igdb import igdb_tool

# Output
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def print_header(text: str):
    print(f"\n{CYAN}{'='*60}")
    print(f"{text.center(60)}")
    print(f"{'='*60}{RESET}")

def save_result(name: str, data):
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"{GREEN}Saved → {path}{RESET}")

@pytest.mark.igdb
def test_01_search_assassins_creed():
    print_header("TEST 1: Search 'Assassin's Creed Valhalla'")
    results = igdb_tool.search_games("Assassin's Creed Valhalla", limit=1)
    assert len(results) > 0
    game = results[0]
    assert game["name"] == "Assassin's Creed Valhalla"
    assert game["id"] == 133004
    save_result("01_valhalla_search", results)
    print(f"{GREEN}EXACT MATCH FOUND! ID: 133004{RESET}")

@pytest.mark.igdb
def test_02_partial_search():
    print_header("TEST 2: Partial search 'cyberpunk'")
    results = igdb_tool.search_games("cyberpunk", limit=5)  # increased limit
    assert len(results) >= 1
    names = [g["name"].lower() for g in results]
    assert any("cyberpunk" in name for name in names)
    save_result("02_cyberpunk", results)
    print(f"{GREEN}Partial match success: {[g['name'] for g in results]}{RESET}")

@pytest.mark.igdb
def test_03_fallback_keyword():
    print_header("TEST 3: Fallback 'odyssey'")
    results = igdb_tool.search_games("assassin odyssey", limit=1)
    assert len(results) > 0
    name = results[0]["name"]
    assert "Odyssey" in name
    save_result("03_odyssey", results)
    print(f"{GREEN}Fallback worked → {name}{RESET}")

@pytest.mark.igdb
def test_04_recent_games():
    print_header("TEST 4: Recent games")
    games = igdb_tool.get_recent_games(limit=5)
    assert len(games) == 5
    save_result("04_recent", games)
    print(f"{GREEN}Got 5 recent games:{RESET}")
    for g in games[:3]:
        date = datetime.fromtimestamp(g["first_release_date"], tz=UTC).strftime("%Y-%m-%d")
        print(f"   • {g['name']} ({date})")

@pytest.mark.igdb
def test_05_valhalla_dlcs():
    print_header("TEST 5: Valhalla + Dawn of Ragnarök")
    game = igdb_tool.get_game_by_id(133004)
    assert game["name"] == "Assassin's Creed Valhalla"
    save_result("05_valhalla_full", game)

    dlcs = igdb_tool.get_expansions_and_dlcs(133004)
    save_result("05_dlcs", dlcs)
    print(f"{GREEN}Found {len(dlcs)} DLCs/Expansions including Ragnarök (Unicode fixed)!{RESET}")
    ragnarok = [d for d in dlcs if "ragnarok" in d["name"].lower().replace("ö", "o")]
    assert len(dlcs) > 0 and any("ragnarok" in d["name"].lower().replace("ö", "o") for d in dlcs)
    print(f"{GREEN}Found Dawn of Ragnarök + {len(dlcs)} total DLCs!{RESET}")

@pytest.mark.igdb
def test_06_fetch_both():
    print_header("TEST 6: fetch_both()")
    data = igdb_tool.fetch_both("GTA VI", limit=6)
    assert len(data["recent_games"]) > 0
    assert len(data["searched_games"]) > 0 or "GTA" in " ".join(g["name"] for g in data["recent_games"])
    save_result("06_fetch_both", data)
    print(f"{GREEN}fetch_both() worked perfectly!{RESET}")

if __name__ == "__main__":
    print(f"{YELLOW}IGDB TEST SUITE – NOV 2025{RESET}")
    print(f"{YELLOW}Make sure .env has TWITCH_CLIENT_ID & TWITCH_CLIENT_SECRET{RESET}\n")
    pytest.main(["-s", __file__])