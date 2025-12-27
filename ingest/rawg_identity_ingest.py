"""
RAWG Canonical Identity Ingestion Pipeline

Fetches, cleans, transforms, and validates a Game identity
before it is eligible for persistence or vectorization.
"""

import argparse
from typing import Any, Dict, Optional

from data.rawg_data import fetch_rawg_game_data
from pre_process.cleaner import RAWGCleaner
from ingest.rawg_identity_validator import validate_game_identity


# ------------------------------------------------------------------
# Minimal local helpers (replicated intentionally for independence)
# ------------------------------------------------------------------

def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_iso_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return value[:10]
    except Exception:
        return None


# ------------------------------------------------------------------
# Canonical Game transformer (identity-focused only)
# ------------------------------------------------------------------

def _create_game_object(source_data: Dict[str, Any]) -> Dict[str, Any]:
    ratings = source_data.get("ratings", {})
    source = source_data.get("source", {})
    platforms = source_data.get("platforms") or []

    return {
        "game_id": _safe_int(source_data.get("game_id")),
        "title": source_data.get("title"),
        "description": source_data.get("description"),
        "release_date": _safe_iso_date(source_data.get("release_date")),
        "release_year": _safe_int(source_data.get("release_year")),
        "genres": source_data.get("genres") or [],
        "developers": source_data.get("developers") or [],
        "publishers": source_data.get("publishers") or [],
        "tags": source_data.get("tags") or [],
        "average_rating": _safe_float(ratings.get("average_rating")),
        "metacritic_score": _safe_int(ratings.get("metacritic")),
        "source_rawg_url": source.get("rawg_url"),
        "last_updated": _safe_iso_date(source.get("last_updated")),
        "has_platform_specs": bool(platforms),
    }


# ------------------------------------------------------------------
# Orchestration function
# ------------------------------------------------------------------

def fetch_and_prepare_identity(game_name: str) -> Dict[str, Any]:
    """
    End-to-end RAWG identity preparation.

    Steps:
    1. Fetch RAWG data
    2. Clean raw payload
    3. Transform into canonical Game object
    4. Validate against canonical identity rules

    Returns:
        Canonical, validated Game dictionary
    """

    # 1. Fetch
    raw = fetch_rawg_game_data(game_name)

    # 2. Clean
    cleaner = RAWGCleaner()
    cleaned = cleaner.clean(raw)
    if not cleaned:
        raise RuntimeError("RAWG cleaning failed")

    # 3. Transform
    game_obj = _create_game_object(cleaned)

    # 4. Validate
    validate_game_identity(game_obj)

    return game_obj


# ------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAWG Canonical Identity Ingest")
    parser.add_argument("--game", "-g", required=True, help="Game name to fetch")
    args = parser.parse_args()

    game = fetch_and_prepare_identity(args.game)
    print("\n=== Canonical Game Identity ===")
    for k, v in game.items():
        print(f"{k}: {v}")
