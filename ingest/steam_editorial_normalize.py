"""
Steam Editorial Normalization Pipeline

Fetches and normalizes a Steam store page into the same editorial
object shape ingest/gamespot_editorial_normalize.py produces, so
EditorialChunker.process_game_editorial can consume it unchanged.
Strictly linked to a Canonical Game entity.

This module is intentionally limited to normalization only.
No chunking. No embeddings. No vector logic.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid5

from data.steam_data import fetch_steam_game_data

_NS = UUID("12345678-1234-5678-1234-567812345678")


def fetch_and_prepare_steam(
    game_name: str,
    canonical_game_uuid: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch and normalize Steam store editorial content for `game_name`.

    Args:
        game_name: Human-readable game name (used for Steam storesearch)
        canonical_game_uuid: UUID of the canonical Game entity (REQUIRED)

    Returns:
        Editorial payload dict (same shape as fetch_and_prepare_gamespot)
        or None if unavailable.
    """
    if not canonical_game_uuid:
        raise ValueError("canonical_game_uuid is required (No Orphan Rule)")

    data = fetch_steam_game_data(game_name)
    if not data:
        return None

    appid = data.get("_appid")
    if not appid:
        return None

    name = data.get("name") or game_name

    articles: List[Dict[str, Any]] = []
    about = data.get("about_the_game")
    if about:
        articles.append(
            {"title": f"{name} — About", "date": None, "deck": None, "body": about}
        )

    detailed = data.get("detailed_description")
    if detailed and detailed != about:
        articles.append(
            {
                "title": f"{name} — Description",
                "date": None,
                "deck": None,
                "body": detailed,
            }
        )

    if not articles:
        return None

    release_date = (data.get("release_date") or {}).get("date")

    editorial_object = {
        "metadata": {
            "id": appid,
            "name": name,
            "slug": None,
            "release_date": release_date,
        },
        "summary": {
            "deck": data.get("short_description"),
            "description": None,
        },
        "reviews": {"average_score": None, "items": []},
        "articles": articles,
    }

    uuid_seed = f"steam_{appid}"
    steam_uuid = str(uuid5(_NS, uuid_seed))

    return {
        "uuid": steam_uuid,
        "class": "EditorialSource",
        "properties": {
            **editorial_object,
            "game_uuid": canonical_game_uuid,
            "source": "steam",
        },
    }


# ------------------------------------------------------------------
# CLI test harness
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam Editorial Normalization")
    parser.add_argument("--game", "-g", required=True, help="Game name")
    parser.add_argument(
        "--uuid", required=True, help="Canonical Game UUID (dummy allowed for testing)"
    )

    args = parser.parse_args()

    result = fetch_and_prepare_steam(game_name=args.game, canonical_game_uuid=args.uuid)

    print("\n=== Steam Normalized Payload ===")
    print(result)
