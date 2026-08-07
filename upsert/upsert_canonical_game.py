#!/usr/bin/env python3
"""
Canonical Game Anchor Upsert (Qdrant Compatible)

- Uses Qdrant client
- Enforces Canonical Game contract (NOT schema validation)
- Enforces deterministic UUIDs
- Idempotent (no duplicate writes)
- Fails fast on all invalid states
"""

import argparse
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ingest.identity_resolver import GAME_NAMESPACE_UUID, resolve_identity


# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------

GAME_CLASS_NAME = "Game"


# -------------------------------------------------------------------
# CONTRACT VALIDATION (AUTHORITATIVE)
# -------------------------------------------------------------------

def validate_game_contract(game: dict) -> None:
    """
    Enforces the Canonical Game contract.

    This validates *data integrity*, not storage schema.
    Qdrant enforces payload correctness at query time.

    Identity root is `unified_game_id` (provider-independent), not any
    single vendor's primary key — `rawg_id`/`igdb_id` are optional
    provenance fields, present only when that provider resolved this
    game.
    """

    if not isinstance(game, dict):
        raise RuntimeError("Game contract violation: payload is not a dictionary")

    # ---- Required identity fields ----
    if not game.get("unified_game_id") or not isinstance(game["unified_game_id"], str):
        raise RuntimeError("Game contract violation: missing 'unified_game_id'")

    if "title" not in game or not isinstance(game["title"], str) or not game["title"].strip():
        raise RuntimeError("Game contract violation: missing or empty 'title'")

    # ---- Optional sanity checks (non-fatal but defensive) ----
    if "release_year" in game and game["release_year"] is not None:
        if not isinstance(game["release_year"], int):
            raise RuntimeError("Game contract violation: 'release_year' must be int")

    if "genres" in game and not isinstance(game["genres"], list):
        raise RuntimeError("Game contract violation: 'genres' must be list")

    if "developers" in game and not isinstance(game["developers"], list):
        raise RuntimeError("Game contract violation: 'developers' must be list")


# -------------------------------------------------------------------
# QDRANT CLIENT HELPER
# -------------------------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    return QdrantClient(url=url, api_key=api_key or None)


# -------------------------------------------------------------------
# CORE UPSERT LOGIC
# -------------------------------------------------------------------

def upsert_game_anchor(client: QdrantClient, game_name: str) -> str:
    """
    Resolve identity (provider-independent) → Contract Validate →
    Idempotent Upsert.
    """

    # ------------------------------------------------------------
    # 1. Resolve canonical identity (IGDB, then RAWG)
    # ------------------------------------------------------------
    identity = resolve_identity(game_name)

    game_obj = {
        "unified_game_id": identity.unified_game_id,
        "identity_source": identity.identity_source,
        "title": identity.title,
        "release_year": identity.release_year,
        "rawg_id": identity.source_ids.get("rawg"),
        "igdb_id": identity.source_ids.get("igdb"),
    }

    # ------------------------------------------------------------
    # 2. Enforce Canonical Game contract (NOT schema validation)
    # ------------------------------------------------------------
    validate_game_contract(game_obj)

    # ------------------------------------------------------------
    # 3. Deterministic UUID (seeded from unified_game_id, not any
    #    single vendor's primary key — reproducible across providers)
    # ------------------------------------------------------------
    game_uuid = identity.game_uuid

    # ------------------------------------------------------------
    # 4. Idempotency check
    # ------------------------------------------------------------
    existing = client.retrieve(
        collection_name=GAME_CLASS_NAME,
        ids=[game_uuid],
    )

    if existing:
        print(f"⚠️  Game already exists. UUID={game_uuid}")
        return game_uuid

    # ------------------------------------------------------------
    # 5. Insert canonical Game anchor
    # ------------------------------------------------------------
    client.upsert(
        collection_name=GAME_CLASS_NAME,
        points=[
            PointStruct(
                id=game_uuid,
                vector=[0.0],  # metadata-only collection (1-dim dummy)
                payload=game_obj,
            )
        ],
    )

    return game_uuid


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upsert Canonical Game Anchor (Qdrant)"
    )
    parser.add_argument(
        "--game",
        "-g",
        required=True,
        help="Game name to fetch and persist",
    )
    args = parser.parse_args()

    client: QdrantClient | None = None

    try:
        client = get_qdrant_client()
        game_uuid = upsert_game_anchor(client, args.game)
        print(f"✅ Canonical Game Anchor ready → UUID: {game_uuid}")

    except Exception as exc:
        print(f"❌ FAILURE: {exc}", file=sys.stderr)
        sys.exit(1)

    finally:
        if client:
            client.close()


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
