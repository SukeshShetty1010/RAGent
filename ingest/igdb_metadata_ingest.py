"""
IGDB Relational Metadata Ingestion (Zero Embeddings)

Fetches, cleans, and transforms IGDB relational data (main game, expansions,
editions, bundles, DLCs) and prepares Qdrant-ready payloads that are
STRICTLY linked to a Canonical Game entity.

This module:
✔ Produces relational metadata
✔ Enforces No-Orphan Rule
✔ Produces schema-valid payloads
✘ Does NOT persist data
✘ Does NOT embed vectors
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from uuid import UUID, uuid5

from data.igdb_data import fetch_igdb_game_data
from pre_process.cleaner import IGDBCleaner


# ---------------------------------------------------------------------
# Helpers (intentionally minimal and local)
# ---------------------------------------------------------------------

def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Core orchestration function
# ---------------------------------------------------------------------

def fetch_and_prepare_igdb(
    game_title: str,
    canonical_game_uuid: str,
) -> List[Dict[str, Any]]:
    """
    Fetch, clean, and prepare IGDB relational metadata.

    Args:
        game_title: Human-readable game title (used for IGDB fetch)
        canonical_game_uuid: UUID of the canonical Game entity (REQUIRED)

    Returns:
        Flat list of IGDB_Game payloads ready for Weaviate batch ingestion

    Raises:
        ValueError if canonical_game_uuid is missing
        RuntimeError on malformed IGDB payloads
    """

    if not canonical_game_uuid:
        raise ValueError("canonical_game_uuid is required (No Orphan Rule)")

    # --------------------------------------------------
    # 1. Fetch IGDB data
    # --------------------------------------------------
    raw_igdb = fetch_igdb_game_data(game_title)

    if not isinstance(raw_igdb, dict):
        raise RuntimeError("IGDB fetch returned invalid payload")

    records = raw_igdb.get("clean") or []
    if not isinstance(records, list):
        raise RuntimeError("Unexpected IGDB fetch format: 'clean' is not a list")

    # --------------------------------------------------
    # 2. Clean & normalize into relational tables
    # --------------------------------------------------
    cleaner = IGDBCleaner()
    relational_tables = cleaner.clean_batch(records)

    if not isinstance(relational_tables, dict):
        raise RuntimeError("IGDB cleaner returned invalid relational structure")

    # --------------------------------------------------
    # 3. Transform → Weaviate-ready payloads
    # --------------------------------------------------
    payloads: List[Dict[str, Any]] = []

    for entity_category, items in relational_tables.items():
        if not isinstance(items, list):
            continue

        for item in items:
            igdb_id = _safe_int(item.get("id"))
            if igdb_id is None:
                # Defensive: skip malformed IGDB records
                continue

            # --------------------------------------------------
            # Deterministic UUID
            # Seed: igdb_<igdb_id>_<entity_category>
            # --------------------------------------------------
            uuid_seed = f"igdb_{igdb_id}_{entity_category}"
            _NS = UUID("12345678-1234-5678-1234-567812345678")
            igdb_uuid = str(uuid5(_NS, uuid_seed))

            payload = {
                "uuid": igdb_uuid,
                "class": "IGDB_Game",
                "properties": {
                    # -------------------------------
                    # Identity / classification
                    # -------------------------------
                    "entity_category": entity_category,
                    "igdb_id": igdb_id,
                    "checksum": item.get("checksum"),

                    # -------------------------------
                    # Core descriptors
                    # -------------------------------
                    "name": item.get("name"),
                    "summary": item.get("summary"),
                    "storyline": item.get("storyline"),
                    "version_title": item.get("version_title"),

                    # -------------------------------
                    # Relational lineage
                    # -------------------------------
                    "parent_game": _safe_int(item.get("parent_game")),
                    "version_parent": _safe_int(item.get("version_parent")),
                    "game_type": _safe_int(item.get("game_type")),

                    # -------------------------------
                    # Temporal metadata
                    # -------------------------------
                    "first_release_date": item.get("first_release_date"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),

                    # -------------------------------
                    # Ratings
                    # -------------------------------
                    "aggregated_rating": item.get("aggregated_rating"),
                    "total_rating": item.get("total_rating"),

                    # -------------------------------
                    # Relational ID arrays
                    # -------------------------------
                    "genres": item.get("genres", []),
                    "platforms": item.get("platforms", []),
                    "themes": item.get("themes", []),
                    "keywords": item.get("keywords", []),
                    "tags": item.get("tags", []),
                    "involved_companies": item.get("involved_companies", []),
                    "franchises": item.get("franchises", []),
                    "collections": item.get("collections", []),

                    # -------------------------------
                    # Mandatory Canonical Game link
                    # -------------------------------
                    "game_uuid": canonical_game_uuid,
                },
            }

            payloads.append(payload)

    return payloads


# ---------------------------------------------------------------------
# CLI test harness (local verification only)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IGDB Relational Metadata Ingest (Zero Embeddings)"
    )
    parser.add_argument("--game", "-g", required=True, help="Game title")
    parser.add_argument(
        "--uuid",
        required=True,
        help="Canonical Game UUID (dummy allowed for testing)",
    )

    args = parser.parse_args()

    objects = fetch_and_prepare_igdb(
        game_title=args.game,
        canonical_game_uuid=args.uuid,
    )

    print("\n=== IGDB Relational Payloads ===")
    for obj in objects:
        print(obj)
