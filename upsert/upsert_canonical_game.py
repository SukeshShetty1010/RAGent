#!/usr/bin/env python3
"""
Canonical Game Anchor Upsert (Weaviate v4 Compatible)

- Uses REAL Weaviate v4 client
- Enforces Canonical Game contract (NOT Weaviate schema validation)
- Enforces deterministic UUIDs
- Idempotent (no duplicate writes)
- Fails fast on all invalid states
"""

import argparse
import sys
from uuid import UUID

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.util import generate_uuid5

from ingest.rawg_identity_ingest import fetch_and_prepare_identity


# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------

GAME_NAMESPACE_UUID = UUID("12345678-1234-5678-1234-567812345678")
GAME_CLASS_NAME = "Game"


# -------------------------------------------------------------------
# CONTRACT VALIDATION (AUTHORITATIVE)
# -------------------------------------------------------------------

def validate_game_contract(game: dict) -> None:
    """
    Enforces the Canonical Game contract.

    This validates *data integrity*, not storage schema.
    Weaviate enforces schema correctness at persistence time.
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
# WEAVIATE CLIENT
# -------------------------------------------------------------------

def get_weaviate_client() -> WeaviateClient:
    try:
        client = WeaviateClient(
            connection_params=ConnectionParams.from_url(
                "http://localhost:8080",
                grpc_port=50051,
            )
        )
        client.connect()
        return client
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Weaviate: {exc}") from exc


# -------------------------------------------------------------------
# CORE UPSERT LOGIC
# -------------------------------------------------------------------

def upsert_game_anchor(client: WeaviateClient, game_name: str) -> str:
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
    game_uuid = generate_uuid5(GAME_NAMESPACE_UUID, str(rawg_id))

    # ------------------------------------------------------------
    # 4. Idempotency check
    # ------------------------------------------------------------
    collection = client.collections.get(GAME_CLASS_NAME)

    if collection.data.exists(uuid=game_uuid):
        print(f"⚠️  Game already exists. UUID={game_uuid}")
        return str(game_uuid)

    # ------------------------------------------------------------
    # 5. Insert canonical Game anchor
    # ------------------------------------------------------------
    collection.data.insert(
        uuid=game_uuid,
        properties=game_obj,
    )

    return str(game_uuid)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upsert Canonical Game Anchor (Weaviate v4)"
    )
    parser.add_argument(
        "--game",
        "-g",
        required=True,
        help="Game name to fetch and persist",
    )
    args = parser.parse_args()

    client: WeaviateClient | None = None

    try:
        client = get_weaviate_client()
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
