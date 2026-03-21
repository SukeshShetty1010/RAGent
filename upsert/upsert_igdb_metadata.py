from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ingest.igdb_metadata_ingest import fetch_and_prepare_igdb


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core Upsert Function (Qdrant)
# ------------------------------------------------------------------
def upsert_igdb_context(
    client: QdrantClient,
    game_title: str,
    game_uuid: str,
) -> int:
    """
    Fetch and upsert IGDB relational metadata using Qdrant.

    Design:
    - Zero embeddings (metadata-only)
    - Deterministic UUIDs
    - Idempotent inserts
    - Soft-failure per object
    """

    # --------------------------------------------------
    # 1. Fetch + Prepare IGDB Payloads
    # --------------------------------------------------
    try:
        payloads = fetch_and_prepare_igdb(
            game_title=game_title,
            canonical_game_uuid=game_uuid,
        )
    except Exception as exc:
        logger.warning(
            "IGDB fetch failed for '%s': %s. Skipping IGDB enrichment.",
            game_title,
            exc,
        )
        return 0

    if not payloads:
        logger.info("No IGDB context found for '%s'.", game_title)
        return 0

    # --------------------------------------------------
    # 2. Idempotent Upsert Loop (Qdrant)
    # --------------------------------------------------
    success_count = 0

    for obj in payloads:
        try:
            # ---- Idempotency gate ----
            existing = client.retrieve(
                collection_name="IGDB_Game",
                ids=[obj["uuid"]],
            )

            if existing:
                logger.info(
                    "IGDB entity already exists (UUID=%s). Skipping.",
                    obj["uuid"],
                )
                continue

            # ---- Prepare payload (strip beacon refs) ----
            properties: Dict = dict(obj["properties"])
            game_ref = properties.pop("game", None)

            # Store game_uuid as a payload field instead of a beacon reference
            properties["game_uuid"] = game_uuid

            # ---- Insert ----
            client.upsert(
                collection_name="IGDB_Game",
                points=[
                    PointStruct(
                        id=obj["uuid"],
                        vector=[0.0],  # metadata-only collection (1-dim dummy)
                        payload=properties,
                    )
                ],
            )

            success_count += 1

        except Exception as exc:
            logger.warning(
                "Failed to upsert IGDB entity (UUID=%s): %s",
                obj.get("uuid"),
                exc,
            )

    return success_count


# ------------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upsert IGDB relational metadata (Qdrant, zero vectors)"
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Game title (used for IGDB lookup)",
    )
    parser.add_argument(
        "--uuid",
        required=True,
        help="Canonical Game UUID (must already exist)",
    )

    args = parser.parse_args()

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    client = QdrantClient(url=url, api_key=api_key or None)

    try:
        count = upsert_igdb_context(
            client=client,
            game_title=args.game,
            game_uuid=args.uuid,
        )
        print(f"✅ Upserted {count} IGDB entities")
    finally:
        client.close()
