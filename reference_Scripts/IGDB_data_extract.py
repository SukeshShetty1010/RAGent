"""
rough.py

Lightweight IGDB-only fetch script.

Why import from `data.`?
------------------------
This project follows a modular layout where all external data-source logic
(RAWG, IGDB, GameSpot) lives inside the `data/` package. Importing from
`data.igdb_data` ensures:
- No duplication of API logic
- Shared environment-variable handling
- Consistency with the main loader pipeline

This script intentionally bypasses loader.py and fetches ONLY IGDB data.
"""

import json
import re
import sys

# IMPORTANT: project-root import (same level as loader.py)
from data.igdb_data import fetch_igdb_game_data


def _safe_name(name: str) -> str:
    """Create a filesystem-safe filename component."""
    name = name.strip().lower()
    return re.sub(r"[^\w\-]+", "_", name)


def main() -> None:
    try:
        game_name = input("Enter game name: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    if not game_name:
        print("Error: Game name is required.")
        return

    try:
        print(f"[INFO] Fetching IGDB data for '{game_name}' (IGDB ONLY)...")

        # --- IGDB fetch (no RAWG / GameSpot orchestration here) ---
        result = fetch_igdb_game_data(game_name, strip_visual=True)

        # Prefer resolved name for file naming if available
        resolved_name = result.get("resolved_name") or game_name
        safe_name = _safe_name(resolved_name)

        output_file = f"igdb_only_{safe_name}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] IGDB data saved to '{output_file}'")
        print(f"Resolved name: {resolved_name}")
        print(f"Records returned: {len(result.get('clean', []))}")

    except Exception as e:
        # Covers missing API keys, auth failures, network errors, etc.
        print(f"[ERROR] Failed to fetch IGDB data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
