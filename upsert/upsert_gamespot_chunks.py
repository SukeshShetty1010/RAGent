from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

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
# Core Upsert Function — Stage 4 Container (Qdrant)
# ------------------------------------------------------------------
def upsert_gamespot_container(
    client: QdrantClient,
    game_name: str,
    game_uuid: str,
) -> Optional[str]:
    """
    Fetch raw GameSpot editorial data and upsert it as a single
    GameSpot_Game container object linked to the canonical Game.

    Returns:
        UUID of the upserted GameSpot_Game object,
        or None if no data found.
    """

    # --------------------------------------------------
    # 1. Fetch GameSpot payload (SOURCE + CANONICAL UUID)
    # --------------------------------------------------
    payload = ingest_gamespot(
        game_name,
        game_uuid,
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
    # 3. Remove beacon reference and store as payload field
    # --------------------------------------------------
    properties.pop("game", None)
    properties["game_uuid"] = game_uuid

    # --------------------------------------------------
    # 4. Upsert into Qdrant
    # --------------------------------------------------
    try:
        existing = client.retrieve(
            collection_name="GameSpot_Game",
            ids=[obj_uuid],
        )

        if existing:
            logger.info(
                "GameSpot_Game already exists for '%s'. Skipping.",
                game_name,
            )
            return obj_uuid

        client.upsert(
            collection_name="GameSpot_Game",
            points=[
                PointStruct(
                    id=obj_uuid,
                    vector=[0.0],  # metadata-only collection (1-dim dummy)
                    payload=properties,
                )
            ],
        )

        logger.info(
            "GameSpot_Game container upserted successfully for '%s'.",
            game_name,
        )
        return obj_uuid

    except Exception as exc:
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
        description="Stage 4: Upsert GameSpot editorial container (Qdrant)"
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

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    client = QdrantClient(url=url, api_key=api_key or None)

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
