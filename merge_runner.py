#!/usr/bin/env python3
"""
merge_runner.py

Interactive runner that:
 - asks the game name
 - uses loader.fetch_all_sources(...) to fetch live data from RAWG, IGDB, Gamespot
 - calls merge.merge_records(...) to produce a canonical merged dict
 - writes the merged JSON to disk: <safe_name>_merged.json

This runner intentionally keeps I/O and network usage separate from merge.py.
"""
from __future__ import annotations

import argparse
import json
import os
import datetime
from typing import Any, Dict

# Import loader function(s) from your loader.py (assumes loader.py is in project root)
from ingest.loader import fetch_all_sources  # loader.py defines fetch_all_sources(game_name, strip_visual=True)
from ingest.merge import merge_records, validate_merged, safe_name  # local module created above

def write_json(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def main():
    import dotenv
    dotenv.load_dotenv()  # load .env so loader can use API keys

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", help="Game name to fetch and merge")
    parser.add_argument("--outdir", "-o", default=".", help="Output directory for merged JSON")
    parser.add_argument("--no-strip-visual", action="store_true", help="If set, will request loader to NOT strip visual fields")
    args = parser.parse_args()

    game = args.game
    if not game:
        try:
            game = input("Enter the game name to fetch live: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return
    if not game:
        print("Game name required.")
        return

    outdir = args.outdir or "."
    os.makedirs(outdir, exist_ok=True)

    print(f"[{datetime.datetime.utcnow().isoformat()}Z] Fetching live data for '{game}' using loader.fetch_all_sources()...")
    # loader.fetch_all_sources returns a dict like {"rawg": [...], "igdb": [...], "gamespot": [...]} by design. See loader.py. :contentReference[oaicite:5]{index=5}
    # pass strip_visual depending on flag inverse semantics
    strip_visual = not args.no_strip_visual
    results = fetch_all_sources(game, strip_visual=strip_visual)

    print("Normalizing & merging...")
    merged = merge_records(results)

    # Validate merged and print warnings if any
    ok, warnings = validate_merged(merged)
    if not ok:
        print("Warnings during validation:")
        for w in warnings:
            print(" -", w)

    # Write merged output
    base = merged.get("slug") or safe_name(game)
    out_path = os.path.join(outdir, f"{base}_merged.json")
    write_json(merged, out_path)
    print("Merged file written:", out_path)

if __name__ == "__main__":
    main()
