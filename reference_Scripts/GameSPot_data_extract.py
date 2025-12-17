"""
rough.py

Standalone GameSpot-only fetch script.

This script intentionally isolates GameSpot logic from loader.py.
It imports and uses `fetch_gamespot_data` directly from `data.gamespot_data`
and does NOT trigger RAWG or IGDB orchestration logic from loader.py.

Notes:
- GameSpot saving is handled internally by fetch_gamespot_data (save=True).
- API keys are read from environment variables as designed in gamespot_data.py.
"""

import sys

# Strictly required import path (project-root level script)
from data.gamespot_data import fetch_gamespot_data


def main() -> None:
    try:
        game_name = input("Enter game name: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    if not game_name:
        print("❌ Error: Game name is required.")
        return

    try:
        print(f"[INFO] Fetching GameSpot data for '{game_name}' (GameSpot ONLY)...")

        # Call GameSpot fetcher
        # - save=True ensures the JSON is written internally
        # - output_dir='.' keeps behavior consistent with loader.py
        result = fetch_gamespot_data(
            game_name,
            save=True,
            output_dir=".",
        )

        # fetch_gamespot_data prints the exact save path internally,
        # so we only provide a confirmation here.
        games_count = result.get("games_count")
        print(f"[SUCCESS] GameSpot fetch complete. Games found: {games_count}")

    except Exception as e:
        # Covers missing API key, network errors, HTTP errors, etc.
        print(f"[ERROR] Failed to fetch GameSpot data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
