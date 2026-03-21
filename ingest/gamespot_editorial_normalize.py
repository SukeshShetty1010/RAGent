"""
GameSpot Editorial Normalization Pipeline

Fetches, cleans, and normalizes GameSpot editorial content into a
single structured GameSpot_Game object, strictly linked to a
Canonical Game entity.

This module is intentionally limited to normalization only.
No chunking. No embeddings. No vector logic.
"""

import argparse
from typing import Any, Dict, Optional

from uuid import UUID, uuid5

from data.gamespot_data import fetch_gamespot_data
from pre_process.cleaner import GameSpotCleaner


# ------------------------------------------------------------------
# Minimal helpers (replicated to preserve isolation)
# ------------------------------------------------------------------

def _safe_iso_date(value: Any) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return value[: len(fmt)]
        except Exception:
            continue

    return None


# ------------------------------------------------------------------
# Transformation logic (replicated from merge.py exactly)
# ------------------------------------------------------------------

def _create_gamespot_object(source_data: Dict[str, Any]) -> Dict[str, Any]:
    metadata = source_data.get("metadata", {}) or {}
    summary = source_data.get("summary", {}) or {}
    reviews = source_data.get("reviews", {}) or {}
    articles = source_data.get("articles", []) or []

    return {
        "metadata": {
            "id": metadata.get("id"),
            "name": metadata.get("name"),
            "slug": metadata.get("slug"),
            "release_date": _safe_iso_date(metadata.get("release_date")),
        },
        "summary": {
            "deck": summary.get("deck"),
            "description": summary.get("description"),
        },
        "reviews": {
            "average_score": reviews.get("average_score"),
            "items": [
                {
                    "title": r.get("title"),
                    "score": r.get("score"),
                    "date": _safe_iso_date(r.get("date")),
                    "body": r.get("body"),
                }
                for r in (reviews.get("items") or [])
                if isinstance(r, dict)
            ],
        },
        "articles": [
            {
                "title": a.get("title"),
                "date": _safe_iso_date(a.get("date")),
                "deck": a.get("deck"),
                "body": a.get("body"),
            }
            for a in articles
            if isinstance(a, dict)
        ],
    }


# ------------------------------------------------------------------
# Orchestration function
# ------------------------------------------------------------------

def fetch_and_prepare_gamespot(
    game_name: str,
    canonical_game_uuid: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch, clean, normalize, and link GameSpot editorial data.

    Args:
        game_name: Human-readable game name
        canonical_game_uuid: UUID of the canonical Game entity (REQUIRED)

    Returns:
        Single GameSpot_Game payload dictionary or None if unavailable
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
    # 3. Transform
    # --------------------------------------------------
    gamespot_obj = _create_gamespot_object(cleaned)

    gamespot_id = gamespot_obj.get("metadata", {}).get("id")
    if not gamespot_id:
        # Without a stable GameSpot ID we cannot create a deterministic UUID
        return None

    # --------------------------------------------------
    # 4. Deterministic UUID
    # Seed: gamespot_<gamespot_id>
    # --------------------------------------------------
    uuid_seed = f"gamespot_{gamespot_id}"
    _NS = UUID("12345678-1234-5678-1234-567812345678")
    gamespot_uuid = str(uuid5(_NS, uuid_seed))

    # --------------------------------------------------
    # 5. Inject Canonical Game Reference
    # --------------------------------------------------
    payload = {
        "uuid": gamespot_uuid,
        "class": "GameSpot_Game",
        "properties": {
            **gamespot_obj,
            "game_uuid": canonical_game_uuid,
        },
    }

    return payload


# ------------------------------------------------------------------
# CLI test harness
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GameSpot Editorial Normalization"
    )
    parser.add_argument("--game", "-g", required=True, help="Game name")
    parser.add_argument(
        "--uuid",
        required=True,
        help="Canonical Game UUID (dummy allowed for testing)",
    )

    args = parser.parse_args()

    result = fetch_and_prepare_gamespot(
        game_name=args.game,
        canonical_game_uuid=args.uuid,
    )

    print("\n=== GameSpot Normalized Payload ===")
    print(result)
