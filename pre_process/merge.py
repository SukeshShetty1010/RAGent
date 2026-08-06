# merge.py
"""
Transforms cleaned RAWG data into flat objects matching:
- Game_Schema.json
- PlatformSpec_Schema.json

Additionally:
- Transforms IGDB relational output into a flat list matching IGDB_Schema.json
- Transforms GameSpot cleaned output into objects matching GameSpot_Schema.json

Outputs the resulting objects to stdout for verification.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------
# Helpers (PRESERVE EXACTLY)
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
# RAWG Core transformers (DO NOT MODIFY)
# ---------------------------------------------------------
def create_game_object(source_data: Dict[str, Any]) -> Dict[str, Any]:
    ratings = source_data.get("ratings", {})
    source = source_data.get("source", {})
    platforms = source_data.get("platforms") or []

    return {
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


def create_platform_objects(source_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    game_id = safe_int(source_data.get("game_id"))
    platforms = source_data.get("platforms") or []

    out: List[Dict[str, Any]] = []
    for p in platforms:
        out.append(
            {
                "platform_name": p.get("platform_name"),
                "platform_family": p.get("platform_family"),
                "release_date": safe_iso_date(p.get("release_date")),
                "requirements_minimum": p.get("requirements_minimum"),
                "requirements_recommended": p.get("requirements_recommended"),
                "game": game_id,  # placeholder reference
            }
        )
    return out


# ---------------------------------------------------------
# IGDB Transformer (DO NOT MODIFY)
# ---------------------------------------------------------
def create_igdb_objects(source_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    igdb_objects: List[Dict[str, Any]] = []

    for category, items in source_data.items():
        if not isinstance(items, list):
            continue

        for item in items:
            igdb_objects.append(
                {
                    "entity_category": category,
                    "id": safe_int(item.get("id")),
                    "checksum": item.get("checksum"),
                    "name": item.get("name"),
                    "summary": item.get("summary"),
                    "storyline": item.get("storyline"),
                    "version_title": item.get("version_title"),
                    "parent_game": safe_int(item.get("parent_game")),
                    "version_parent": safe_int(item.get("version_parent")),
                    "game_type": safe_int(item.get("game_type")),
                    "first_release_date": safe_iso_date(item.get("first_release_date")),
                    "created_at": safe_iso_date(item.get("created_at")),
                    "updated_at": safe_iso_date(item.get("updated_at")),
                    "aggregated_rating": safe_float(item.get("aggregated_rating")),
                    "total_rating": safe_float(item.get("total_rating")),
                    "genres": item.get("genres") or [],
                    "platforms": item.get("platforms") or [],
                    "themes": item.get("themes") or [],
                    "keywords": item.get("keywords") or [],
                    "tags": item.get("tags") or [],
                    "involved_companies": item.get("involved_companies") or [],
                    "franchises": item.get("franchises") or [],
                    "collections": item.get("collections") or [],
                }
            )

    return igdb_objects


# ---------------------------------------------------------
# GameSpot Transformer (NEW)
# ---------------------------------------------------------
def create_gamespot_objects(source_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map cleaned GameSpot payload into GameSpot_Schema.json structure.
    The schema is nested and represented as a single object.
    """
    metadata = source_data.get("metadata", {}) or {}
    summary = source_data.get("summary", {}) or {}
    reviews = source_data.get("reviews", {}) or {}
    articles = source_data.get("articles", []) or []

    return {
        "metadata": {
            "id": safe_int(metadata.get("id")),
            "name": metadata.get("name"),
            "slug": metadata.get("slug"),
            "release_date": safe_iso_date(metadata.get("release_date")),
        },
        "summary": {
            "deck": summary.get("deck"),
            "description": summary.get("description"),
        },
        "reviews": {
            "average_score": safe_float(reviews.get("average_score")),
            "items": [
                {
                    "title": r.get("title"),
                    "score": safe_float(r.get("score")),
                    "date": safe_iso_date(r.get("date")),
                    "body": r.get("body"),
                }
                for r in (reviews.get("items") or [])
                if isinstance(r, dict)
            ],
        },
        "articles": [
            {
                "title": a.get("title"),
                "date": safe_iso_date(a.get("date")),
                "deck": a.get("deck"),
                "body": a.get("body"),
            }
            for a in articles
            if isinstance(a, dict)
        ],
    }


# ---------------------------------------------------------
# CLI / Execution
# ---------------------------------------------------------
def main():
    # ---------- RAWG ----------
    with open("cleaned_rawg_data.json", "r", encoding="utf-8") as f:
        rawg_source = json.load(f)

    game = create_game_object(rawg_source)
    platforms = create_platform_objects(rawg_source)

    print("=== RAWG: Game Object ===")
    print(json.dumps(game, indent=2, ensure_ascii=False))

    print("\n=== RAWG: PlatformSpec Objects ===")
    print(json.dumps(platforms, indent=2, ensure_ascii=False))

    # ---------- IGDB ----------
    with open("igdb_relational_output.json", "r", encoding="utf-8") as f:
        igdb_source = json.load(f)

    igdb_objects = create_igdb_objects(igdb_source)

    print("\n=== IGDB: Unified Objects ===")
    print(json.dumps(igdb_objects, indent=2, ensure_ascii=False))

    # ---------- GameSpot ----------
    with open("cleaned_data_output.json", "r", encoding="utf-8") as f:
        gamespot_source = json.load(f)

    gamespot_object = create_gamespot_objects(gamespot_source)

    print("\n=== GameSpot: Game Object ===")
    print(json.dumps(gamespot_object, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
