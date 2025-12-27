"""
IGDB Relational Metadata Ingestion (Zero Embeddings)

Fetches, cleans, and transforms IGDB relational data (main game, expansions,
editions, bundles, DLCs) and prepares Weaviate-ready payloads that are
STRICTLY linked to a Canonical Game entity.

This module exists to support graph traversal and filtering without polluting
the semantic vector space.
"""

import argparse
from typing import Any, Dict, List

from weaviate.util import generate_uuid5

from data.igdb_data import fetch_igdb_game_data
from pre_process.cleaner import IGDBCleaner


# ---------------------------------------------------------------------
# Helpers (intentionally minimal and local)
# ---------------------------------------------------------------------

def _safe_int(value: Any):
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
        canonical_game_uuid: UUID of the canonical Game object (REQUIRED)

    Returns:
        Flat list of IGDB_Game payloads ready for batch ingestion

    Raises:
        ValueError if canonical_game_uuid is missing
    """

    if not canonical_game_uuid:
        raise ValueError("canonical_game_uuid is required (No Orphan Rule)")

    # --------------------------------------------------
    # 1. Fetch
    # --------------------------------------------------
    raw_igdb = fetch_igdb_game_data(game_title)

    # IGDB fetcher returns a wrapper dict; extract records
    records = raw_igdb.get("clean") or []
    if not isinstance(records, list):
        raise RuntimeError("Unexpected IGDB fetch format")

    # --------------------------------------------------
    # 2. Clean (relational normalization)
    # --------------------------------------------------
    cleaner = IGDBCleaner()
    relational_tables = cleaner.clean_batch(records)

    # --------------------------------------------------
    # 3. Transform + Cross-Link
    # --------------------------------------------------
    payloads: List[Dict[str, Any]] = []

    for entity_category, items in relational_tables.items():
        if not isinstance(items, list):
            continue

        for item in items:
            igdb_id = _safe_int(item.get("id"))
            if igdb_id is None:
                # Defensive: skip malformed IGDB objects
                continue

            # --------------------------------------------------
            # Deterministic UUID
            # Seed: igdb_<igdb_id>_<entity_category>
            # --------------------------------------------------
            uuid_seed = f"igdb_{igdb_id}_{entity_category}"
            igdb_uuid = generate_uuid5(uuid_seed)

            payload = {
                "uuid": igdb_uuid,
                "class": "IGDB_Game",
                "properties": {
                    "entity_category": entity_category,
                    "id": igdb_id,
                    "name": item.get("name"),
                    "summary": item.get("summary"),
                    "storyline": item.get("storyline"),
                    "parent_game": _safe_int(item.get("parent_game")),
                    "version_parent": _safe_int(item.get("version_parent")),
                    # --------------------------------------------------
                    # Hard reference to Canonical Game (MANDATORY)
                    # --------------------------------------------------
                    "game": {
                        "beacon": f"weaviate://localhost/Game/{canonical_game_uuid}"
                    },
                },
            }

            payloads.append(payload)

    return payloads


# ---------------------------------------------------------------------
# CLI test harness
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
