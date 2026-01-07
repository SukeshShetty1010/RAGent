from __future__ import annotations

import argparse
import logging
from typing import Optional

import weaviate
from weaviate import WeaviateClient
from weaviate.exceptions import WeaviateBaseError

from ingest.ingest_gamespot import ingest_gamespot


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core Upsert Function — Stage 4 Container
# ------------------------------------------------------------------
def upsert_gamespot_container(
    client: WeaviateClient,
    game_name: str,
    game_uuid: str,
) -> Optional[str]:
    """
    Fetch raw GameSpot editorial data and upsert it as a single
    GameSpot_Game container object linked to the canonical Game.

    Returns:
        UUID of the upserted GameSpot_Game object, or None if no data found.
    """

    # --------------------------------------------------
    # 1. Fetch GameSpot payload
    # --------------------------------------------------
    payload = ingest_gamespot(
        game_name=game_name,
        game_uuid=game_uuid,
    )

    if payload is None:
        logger.warning("No GameSpot data found for %s", game_name)
        return None

    if not isinstance(payload, dict):
        raise RuntimeError("GameSpot ingest returned invalid payload")

    # --------------------------------------------------
    # 2. Extract UUID and properties
    # --------------------------------------------------
    obj_uuid = payload.get("uuid")
    properties = payload.get("properties", {})

    if not obj_uuid or not isinstance(properties, dict):
        raise RuntimeError("Invalid GameSpot payload structure")

    # --------------------------------------------------
    # 3. Remove V3-style reference (CRITICAL)
    # --------------------------------------------------
    # Payload includes:
    #   "game": {"beacon": "..."}
    # This MUST be removed and passed via `references`
    properties.pop("game", None)

    # --------------------------------------------------
    # 4. Upsert into Weaviate (v4)
    # --------------------------------------------------
    collection = client.collections.get("GameSpot_Game")

    try:
        if collection.data.exists(uuid=obj_uuid):
            logger.info(
                "GameSpot_Game already exists for '%s'. Skipping.",
                game_name,
            )
            return obj_uuid

        collection.data.insert(
            uuid=obj_uuid,
            properties=properties,
            references={
                "game": game_uuid,
            },
        )

        logger.info(
            "GameSpot_Game container upserted successfully for '%s'.",
            game_name,
        )
        return obj_uuid

    except WeaviateBaseError as exc:
        logger.error(
            "Failed to upsert GameSpot_Game for '%s': %s",
            game_name,
            exc,
        )
        raise


# ------------------------------------------------------------------
# CLI Entrypoint (Stage 4 isolation test)
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 4: Upsert GameSpot editorial container"
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Game name (e.g., 'Far Cry 5')",
    )
    parser.add_argument(
        "--uuid",
        required=True,
        help="Canonical Game UUID",
    )

    args = parser.parse_args()

    client = weaviate.connect_to_local()

    try:
        result = upsert_gamespot_container(
            client=client,
            game_name=args.game,
            game_uuid=args.uuid,
        )

        if result:
            print(f"✅ GameSpot_Game upserted. UUID={result}")
        else:
            print("⚠️ No GameSpot data found.")

    finally:
        client.close()
