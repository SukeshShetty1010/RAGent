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
from uuid import UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ingest.rawg_identity_ingest import fetch_and_prepare_identity


# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------

GAME_NAMESPACE_UUID = UUID("12345678-1234-5678-1234-567812345678")
GAME_CLASS_NAME = "Game"


# -------------------------------------------------------------------
# UUID HELPER (replaces weaviate.util.generate_uuid5)
# -------------------------------------------------------------------

def _generate_uuid5(namespace: UUID, seed: str) -> str:
    return str(uuid5(namespace, seed))


# -------------------------------------------------------------------
# CONTRACT VALIDATION (AUTHORITATIVE)
# -------------------------------------------------------------------

def validate_game_contract(game: dict) -> None:
    """
    Enforces the Canonical Game contract.

    This validates *data integrity*, not storage schema.
    Qdrant enforces payload correctness at query time.
    """

    if not isinstance(game, dict):
        raise RuntimeError("Game contract violation: payload is not a dictionary")

    # ---- Required identity fields ----
    if "game_id" not in game or game["game_id"] is None:
        raise RuntimeError("Game contract violation: missing 'game_id'")

    if not isinstance(game["game_id"], int):
        raise RuntimeError("Game contract violation: 'game_id' must be int")

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
    Fetch → Contract Validate → Deterministic UUID → Idempotent Upsert
    """

    # ------------------------------------------------------------
    # 1. Fetch canonical identity from RAWG
    # ------------------------------------------------------------
    game_obj = fetch_and_prepare_identity(game_name)

    # ------------------------------------------------------------
    # 2. Enforce Canonical Game contract (NOT schema validation)
    # ------------------------------------------------------------
    validate_game_contract(game_obj)

    # ------------------------------------------------------------
    # 3. Deterministic UUID (RAWG ID is the identity root)
    # ------------------------------------------------------------
    rawg_id = game_obj["game_id"]
    game_uuid = _generate_uuid5(GAME_NAMESPACE_UUID, str(rawg_id))

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
