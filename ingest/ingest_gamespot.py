"""
GameSpot Editorial Ingestion (NO UPSERT)

Responsibilities:
- Fetch GameSpot data
- Clean it
- Normalize into GameSpot_Game schema
- Generate deterministic UUID
- Attach canonical Game reference
- RETURN the payload (do NOT persist)

This module:
✔ produces data
✘ does NOT store data
✘ does NOT chunk
✘ does NOT embed
"""

from __future__ import annotations

from typing import Dict, Optional

from weaviate.util import generate_uuid5

from data.gamespot_data import fetch_gamespot_data
from pre_process.cleaner import GameSpotCleaner


# ------------------------------------------------------------------
# Core ingestion function
# ------------------------------------------------------------------

def ingest_gamespot(
    game_name: str,
    canonical_game_uuid: str,
) -> Optional[Dict]:
    """
    Fetch, clean, and normalize GameSpot editorial data.

    Args:
        game_name: Human-readable game name
        canonical_game_uuid: UUID of the canonical Game (REQUIRED)

    Returns:
        GameSpot_Game ingestion payload:
        {
            "uuid": str,
            "class": "GameSpot_Game",
            "properties": { ...normalized schema... }
        }

        Returns None if GameSpot data is unavailable.
    """

    if not canonical_game_uuid:
        raise ValueError("canonical_game_uuid is required (No Orphan Rule)")

    # --------------------------------------------------
    # 1. Fetch
    # --------------------------------------------------
    raw = fetch_gamespot_data(game_name)
    if not raw:
        return None

    # --------------------------------------------------
    # 2. Clean
    # --------------------------------------------------
    cleaner = GameSpotCleaner()
    cleaned = cleaner.clean(raw)
    if not cleaned:
        return None

    # --------------------------------------------------
    # 3. Validate required identity
    # --------------------------------------------------
    gamespot_id = cleaned.get("metadata", {}).get("id")
    if not gamespot_id:
        # Cannot create deterministic identity without GameSpot ID
        return None

    # --------------------------------------------------
    # 4. Deterministic UUID
    # --------------------------------------------------
    uuid_seed = f"gamespot_{gamespot_id}"
    gamespot_uuid = generate_uuid5(uuid_seed)

    # --------------------------------------------------
    # 5. Build ingestion payload (NO persistence)
    # --------------------------------------------------
    payload = {
        "uuid": gamespot_uuid,
        "class": "GameSpot_Game",
        "properties": {
            **cleaned,
            "game": {
                "beacon": f"weaviate://localhost/Game/{canonical_game_uuid}"
            },
        },
    }

    return payload


# ------------------------------------------------------------------
# CLI test harness (local verification only)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="GameSpot Ingestion (No Upsert)")
    parser.add_argument("--game", required=True)
    parser.add_argument("--uuid", required=True, help="Canonical Game UUID")
    args = parser.parse_args()

    result = ingest_gamespot(
        game_name=args.game,
        canonical_game_uuid=args.uuid,
    )

    if not result:
        print("No GameSpot data available.")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
