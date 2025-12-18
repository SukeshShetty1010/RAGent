# merge.py
"""
Transforms cleaned RAWG data into flat objects matching:
- Game_Schema.json
- PlatformSpec_Schema.json

Outputs the resulting Game object and PlatformSpec objects to stdout.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def safe_iso_date(value: Optional[str]) -> Optional[str]:
    """
    Parse a date/datetime string and return RFC3339-compatible ISO date.
    If parsing fails or value is None, return None.
    """
    if not value:
        return None

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            dt = datetime.strptime(value[: len(fmt)], fmt)
            return dt.date().isoformat()
        except Exception:
            continue

    try:
        return datetime.fromisoformat(value).date().isoformat()
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


# ---------------------------------------------------------
# Core transformers
# ---------------------------------------------------------
def create_game_object(source_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a flat Game object matching Game_Schema.json
    """
    ratings = source_data.get("ratings", {})
    source = source_data.get("source", {})
    platforms = source_data.get("platforms") or []

    game_obj = {
        "game_id": safe_int(source_data.get("game_id")),
        "title": source_data.get("title"),
        "description": source_data.get("description"),
        "release_date": safe_iso_date(source_data.get("release_date")),
        "release_year": safe_int(source_data.get("release_year")),
        "genres": source_data.get("genres") or [],
        "developers": source_data.get("developers") or [],
        "publishers": source_data.get("publishers") or [],
        "tags": source_data.get("tags") or [],
        "average_rating": safe_float(ratings.get("average_rating")),
        "metacritic_score": safe_int(ratings.get("metacritic")),
        "source_rawg_url": source.get("rawg_url"),
        "last_updated": safe_iso_date(source.get("last_updated")),
        "has_platform_specs": bool(platforms),
    }

    return game_obj


def create_platform_objects(source_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create PlatformSpec objects matching PlatformSpec_Schema.json.
    Uses game_id as a placeholder reference to the Game object.
    """
    game_id = safe_int(source_data.get("game_id"))
    platforms = source_data.get("platforms") or []

    platform_objects: List[Dict[str, Any]] = []

    for p in platforms:
        platform_objects.append(
            {
                "platform_name": p.get("platform_name"),
                "platform_family": p.get("platform_family"),
                "release_date": safe_iso_date(p.get("release_date")),
                "requirements_minimum": p.get("requirements_minimum"),
                "requirements_recommended": p.get("requirements_recommended"),
                # Placeholder cross-ref (to be replaced by Weaviate reference)
                "game": game_id,
            }
        )

    return platform_objects


# ---------------------------------------------------------
# CLI / Execution
# ---------------------------------------------------------
def main():
    with open("cleaned_rawg_data.json", "r", encoding="utf-8") as f:
        source_data = json.load(f)

    game = create_game_object(source_data)
    platforms = create_platform_objects(source_data)

    print("=== Game Object ===")
    print(json.dumps(game, indent=2, ensure_ascii=False))

    print("\n=== PlatformSpec Objects ===")
    print(json.dumps(platforms, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
