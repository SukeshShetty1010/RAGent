#!/usr/bin/env python3
"""
Canonical Game Anchor Upsert (Weaviate v4 Compatible)

- Uses REAL Weaviate v4 client
- Validates payload against JSON Schema
- Enforces deterministic UUIDs
- Idempotent (no duplicate writes)
- Fails fast on all invalid states
"""

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.util import generate_uuid5
from jsonschema import validate, ValidationError

from ingest.rawg_identity_ingest import fetch_and_prepare_identity


# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------

GAME_NAMESPACE_UUID = UUID("12345678-1234-5678-1234-567812345678")
GAME_CLASS_NAME = "Game"
SCHEMA_PATH = Path(__file__).parent / "vector\schemas\rawg_game.schema.json"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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
        raise RuntimeError(f"Failed to connect to Weaviate: {exc}")


# -------------------------------------------------------------------
# CORE UPSERT LOGIC
# -------------------------------------------------------------------

def upsert_game_anchor(client: WeaviateClient, game_name: str) -> str:
    """
    Fetch → Validate → Idempotent Upsert of Canonical Game
    """

    # 1. Fetch canonical identity
    game_obj = fetch_and_prepare_identity(game_name)

    # 2. Schema validation
    schema = load_schema()
    try:
        validate(instance=game_obj, schema=schema)
    except ValidationError as exc:
        raise RuntimeError(f"Schema validation failed: {exc.message}") from exc

    # 3. Deterministic UUID
    rawg_id = game_obj["game_id"]
    game_uuid = generate_uuid5(GAME_NAMESPACE_UUID, str(rawg_id))

    # 4. Idempotency check
    collection = client.collections.get(GAME_CLASS_NAME)
    if collection.data.exists(uuid=game_uuid):
        print(f"⚠️  Game already exists. UUID={game_uuid}")
        return str(game_uuid)

    # 5. Insert
    collection.data.insert(
        uuid=game_uuid,
        properties=game_obj,
    )

    return str(game_uuid)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
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

    try:
        client = get_weaviate_client()
        uuid = upsert_game_anchor(client, args.game)
        print(f"✅ Canonical Game Anchor ready → UUID: {uuid}")
    except Exception as exc:
        print(f"❌ FAILURE: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
