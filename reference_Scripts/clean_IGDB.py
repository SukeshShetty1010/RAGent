"""
IGDB Relational Transformation Script
====================================

Transforms a flat IGDB `clean` array into a relational dictionary structure
based on classification and normalization rules.

- Classifies entities (Base Game, Expansion, Edition, Bundle, Update)
- Converts Unix timestamps to ISO-8601
- Preserves foreign-key ID arrays
- Preserves checksum for integrity
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def unix_to_iso(ts: Optional[int]) -> Optional[str]:
    """Convert Unix epoch (seconds) to ISO-8601 UTC string."""
    if not ts:
        return None
    try:
        return datetime.utcfromtimestamp(ts).isoformat() + "Z"
    except Exception:
        return None


def normalize_text(text: Optional[str]) -> Optional[str]:
    """Normalize text fields (trim, normalize newlines)."""
    if not text or not isinstance(text, str):
        return None
    return "\n".join(line.strip() for line in text.strip().splitlines())


# ---------------------------------------------------------
# Classification Logic
# ---------------------------------------------------------
def classify_entity(item: Dict[str, Any]) -> str:
    """
    Classify IGDB entity into a target table.

    Mapping logic:
    - game_type == 0 and no parent/version -> Main_Game
    - game_type == 0 and version_parent -> Edition
    - game_type in (1, 2) and parent_game -> Expansion
    - game_type == 14 and parent_game -> Update
    - game_type == 3 -> Bundle
    - game_type == 4 -> Standalone_Experience
    """
    game_type = item.get("game_type")
    parent_game = item.get("parent_game")
    version_parent = item.get("version_parent")

    if game_type == 0 and not parent_game and not version_parent:
        return "Main_Game_Table"
    if game_type == 0 and version_parent:
        return "Edition_Table"
    if game_type in (1, 2) and parent_game:
        return "Expansion_Table"
    if game_type == 14 and parent_game:
        return "Update_Table"
    if game_type == 3:
        return "Bundle_Table"
    if game_type == 4:
        return "Standalone_Experience_Table"

    return "Unclassified"


# ---------------------------------------------------------
# Transformation Logic
# ---------------------------------------------------------
def transform_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single IGDB record according to schema rules."""
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "parent_game": item.get("parent_game"),
        "version_parent": item.get("version_parent"),
        "version_title": item.get("version_title"),
        "game_type": item.get("game_type"),
        "first_release_date": unix_to_iso(item.get("first_release_date")),
        "created_at": unix_to_iso(item.get("created_at")),
        "updated_at": unix_to_iso(item.get("updated_at")),
        "summary": normalize_text(item.get("summary")),
        "storyline": normalize_text(item.get("storyline")),
        "aggregated_rating": item.get("aggregated_rating"),
        "total_rating": item.get("total_rating"),
        "checksum": item.get("checksum"),

        # Foreign-key arrays (preserved as-is)
        "genres": item.get("genres", []),
        "platforms": item.get("platforms", []),
        "themes": item.get("themes", []),
        "keywords": item.get("keywords", []),
        "tags": item.get("tags", []),
        "involved_companies": item.get("involved_companies", []),
        "franchises": item.get("franchises", []),
        "collections": item.get("collections", []),
    }


# ---------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------
def build_relational_structure(clean_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    output = {
        "Main_Game_Table": [],
        "Expansion_Table": [],
        "Edition_Table": [],
        "Bundle_Table": [],
        "Update_Table": [],
        "Standalone_Experience_Table": [],
        "Unclassified": [],
    }

    for item in clean_items:
        table = classify_entity(item)
        transformed = transform_item(item)
        output[table].append(transformed)

    return output


# ---------------------------------------------------------
# Entrypoint (no hardcoded paths)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example usage:
    # Load input JSON from stdin or a file
    with open("igdb_only_assassin_s_creed_valhalla.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    clean_items = raw.get("clean", [])
    relational_output = build_relational_structure(clean_items)

    # Write output
    with open("igdb_relational_output.json", "w", encoding="utf-8") as f:
        json.dump(relational_output, f, indent=2, ensure_ascii=False)

    print("Relational IGDB JSON written to igdb_relational_output.json")
