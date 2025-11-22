import json
import os
from data.gamespot_data import is_record_for_game, GameSpotData

# Small unit tests for the filtering logic.
def load_fixture(name):
    base = os.path.dirname(__file__) or "."
    # the rough run created merged_inputs_docs.json; use that as a fixture if present
    fixtures = [
        os.path.join(os.getcwd(), "merged_inputs_docs.json"),
        os.path.join(os.getcwd(), "all_docs.json"),
        os.path.join(base, "fixtures", name)
    ]
    for p in fixtures:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data
    return None

def test_is_record_for_game_basic():
    # Simple records to test matching heuristics
    game_id = 107438
    game_title = "Far Cry 3"
    game_slug = "far-cry-3-2012"

    # record with explicit id-like field
    rec1 = {"game_id": 107438, "name": "Far Cry 3 - Deluxe"}
    assert is_record_for_game(rec1, game_id, game_title, game_slug)

    # record with slug in site_detail_url
    rec2 = {"site_detail_url": "https://www.gamespot.com/far-cry-3-2012/some-release"}
    assert is_record_for_game(rec2, game_id, game_title, game_slug)

    # record with title containing game title
    rec3 = {"name": "Far Cry 3: Collector's Edition"}
    assert is_record_for_game(rec3, game_id, game_title, game_slug)

    # unrelated record should return False
    rec4 = {"name": "Command & Conquer (1995) Release"}
    assert not is_record_for_game(rec4, game_id, game_title, game_slug)

def test_gamespot_data_filter_integration():
    # If merged_inputs_docs.json exists (created by your rough run), run the integration smoke test.
    fixture = load_fixture("merged_inputs_docs.json")
    if fixture is None:
        # No fixture available; skip this integration test.
        return

    # The fixture items are structured content docs. We'll locate some gamespot items and test filtering logic
    gamespot_items = [d for d in fixture if d.get("metadata", {}).get("source") == "gamespot"]
    assert len(gamespot_items) > 0

    sample = gamespot_items[0]
    unified = sample["metadata"]["unified_game_id"]
    # use GameSpotData.is_record_for_game via the module to test a few records
    # This is a smoke check: ensure that filter doesn't keep obviously wrong releases
    from data.gamespot_data import is_record_for_game
    # check a known bad release (Command & Conquer should not match Far Cry 3)
    bad = {"name": "Command & Conquer", "site_detail_url": "https://www.gamespot.com/command-and-conquer/"}
    assert not is_record_for_game(bad, sample["metadata"]["game_id"], sample["metadata"]["title"], sample["metadata"]["slug"])
