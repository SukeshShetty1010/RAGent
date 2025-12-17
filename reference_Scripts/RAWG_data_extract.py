#!/usr/bin/env python3
"""
rough.py

Stage-0 RAWG-only extractor.

Purpose:
- Fetch RAWG data in complete isolation (no IGDB, no GameSpot, no merging).
- Persist the *raw-but-pruned* RAWG payload as a clean artifact.
- Serve as the first step in a staged ETL pipeline.

Output:
- rawg_only_<safe_game_name>.json
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

# --- Local imports (same directory assumption) ---
from data.rawg_data import fetch_rawg_game_data
from ingest.loader import _safe_name, _write_json


def fetch_and_save_rawg(
    game_name: str,
    outdir: str = ".",
) -> str:
    """
    Fetch RAWG-only data for a game and save it as an isolated JSON artifact.

    Returns:
        Path to the saved JSON file.
    """
    if not game_name or not isinstance(game_name, str):
        raise ValueError("game_name must be a non-empty string")

    print(f"[INFO] Fetching RAWG data only for: '{game_name}'")

    # --- RAWG fetch (no merging, no wrappers) ---
    rawg_data = fetch_rawg_game_data(
        query=game_name,
        strip_visual=True,
        prefer_exact_match=True,
    )

    # --- Deterministic filename ---
    safe = _safe_name(game_name)
    filename = f"rawg_only_{safe}.json"
    out_path = f"{outdir.rstrip('/')}/{filename}"

    # --- Persist using shared helper ---
    _write_json(rawg_data, out_path)

    print(f"[SUCCESS] RAWG-only data saved to: {out_path}")
    return out_path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rough.py",
        description="Stage-0 RAWG-only extractor (no merging).",
    )
    parser.add_argument(
        "--game",
        "-g",
        default="Assassin's Creed Valhalla",
        help="Game name to fetch (default: Assassin's Creed Valhalla)",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default=".",
        help="Output directory (default: current directory)",
    )

    args = parser.parse_args(argv)

    try:
        fetch_and_save_rawg(
            game_name=args.game.strip(),
            outdir=args.outdir,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
