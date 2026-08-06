"""
loader.py

Fetches game data from RAWG, IGDB, and GameSpot using the helper modules
located inside the project's data/ directory.

In __main__:
 - prompts for game name (unless --game is passed)
 - fetches from all 3 sources
 - saves per-source JSON files
 - saves a combined all-sources JSON file
"""

import argparse
import datetime
import json
import logging
import os
import re

# ⬇️ Correct project-based imports
from data.rawg_data import fetch_rawg_game_data
from data.igdb_data import fetch_igdb_game_data
from data.gamespot_data import fetch_gamespot_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _safe_name(name: str) -> str:
    """Make a safe filename from the game name."""
    name = name.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", name) or "game"


def _write_json(obj, filename):
    tmp = filename + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, filename)
    logger.info(f"Saved {filename}")


# -------------------------------------------------------------
# Fetching functions
# -------------------------------------------------------------
def fetch_rawg(game_name: str, strip_visual=True):
    logger.info(f"Fetching RAWG for '{game_name}'...")
    res = fetch_rawg_game_data(game_name, strip_visual=strip_visual)
    return res if isinstance(res, list) else [res]


def fetch_igdb(game_name: str, strip_visual=True):
    logger.info(f"Fetching IGDB for '{game_name}'...")
    res = fetch_igdb_game_data(game_name, strip_visual=strip_visual)
    return res if isinstance(res, list) else [res]


def fetch_gamespot(game_name: str, strip_visual=True):
    logger.info(f"Fetching GameSpot for '{game_name}'...")
    # NOTE: fetch_gamespot_data(save=True by default) will create
    # <safe_name>_gamespot_full_textual.json itself.
    res = fetch_gamespot_data(game_name, strip_visual=strip_visual)
    return res if isinstance(res, list) else [res]


def fetch_all_sources(game_name: str, strip_visual=True):
    return {
        "rawg": fetch_rawg(game_name, strip_visual),
        "igdb": fetch_igdb(game_name, strip_visual),
        "gamespot": fetch_gamespot(game_name, strip_visual),
    }


# -------------------------------------------------------------
# Save outputs
# -------------------------------------------------------------
def save_source_files(game_name: str, results: dict, outdir: str = "."):
    safe = _safe_name(game_name)
    saved = []

    for source, data in results.items():
        # 🔴 Do NOT save a separate GameSpot file here.
        # gamespot_data.fetch_gamespot_data() already creates
        # <safe_name>_gamespot_full_textual.json on its own.
        if source == "gamespot":
            continue

        filename = os.path.join(outdir, f"{safe}_{source}.json")
        payload = {
            "game": game_name,
            "source": source,
            "fetched_at": _now_iso(),
            "count": len(data),
            "records": data,
        }
        _write_json(payload, filename)
        saved.append(filename)

    # combined file (still includes gamespot data in the JSON structure)
    combined = os.path.join(outdir, f"{safe}_all_sources.json")
    combined_payload = {
        "game": game_name,
        "fetched_at": _now_iso(),
        "sources": {k: len(v) for k, v in results.items()},
        "data": results,
    }
    _write_json(combined_payload, combined)
    saved.append(combined)

    return saved


# -------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", help="Name of the game")
    parser.add_argument("--outdir", "-o", default=".", help="Directory to save JSON files")
    parser.add_argument("--no-strip-visual", dest="strip_visual", action="store_false")
    args = parser.parse_args()

    game = args.game
    if not game:
        try:
            game = input("Enter game name to fetch: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    if not game:
        print("Game name required.")
        return

    os.makedirs(args.outdir, exist_ok=True)

    logger.info(f"Fetching data for '{game}'...")
    results = fetch_all_sources(game, strip_visual=args.strip_visual)

    saved_files = save_source_files(game, results, outdir=args.outdir)

    print("\nSaved files:")
    for f in saved_files:
        print(" -", f)


if __name__ == "__main__":
    main()
