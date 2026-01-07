from __future__ import annotations

import argparse
import logging
from typing import Dict, Optional

import weaviate
from weaviate.exceptions import WeaviateBaseError

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
# Helpers
# ------------------------------------------------------------------
def _uuid_from_beacon(beacon: Optional[str]) -> Optional[str]:
    """
    Extract UUID from a Weaviate beacon:
      weaviate://localhost/Game/<uuid> → <uuid>
    """
    if not beacon or not isinstance(beacon, str):
        return None
    try:
        return beacon.rstrip("/").split("/")[-1]
    except Exception:
        return None


# ------------------------------------------------------------------
# Core Upsert Function (Weaviate v4)
# ------------------------------------------------------------------
def upsert_igdb_context(
    client: weaviate.WeaviateClient,
    game_title: str,
    game_uuid: str,
) -> int:
    """
    Fetch and upsert IGDB relational metadata using Weaviate v4.

    Design:
    - Zero embeddings
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
    # 2. Get Collection (Weaviate v4)
    # --------------------------------------------------
    collection = client.collections.get("IGDB_Game")

    success_count = 0

    # --------------------------------------------------
    # 3. Idempotent Upsert Loop
    # --------------------------------------------------
    for obj in payloads:
        try:
            # ---- Idempotency gate ----
            if collection.data.exists(uuid=obj["uuid"]):
                logger.info(
                    "IGDB entity already exists (UUID=%s). Skipping.",
                    obj["uuid"],
                )
                continue

            # ---- Separate properties and references (v4 rule) ----
            properties: Dict = dict(obj["properties"])
            game_ref = properties.pop("game", None)

            references: Dict[str, str] = {}

            if isinstance(game_ref, dict):
                game_uuid_ref = _uuid_from_beacon(game_ref.get("beacon"))
                if game_uuid_ref:
                    references["game"] = game_uuid_ref

            # ---- Insert ----
            collection.data.insert(
                uuid=obj["uuid"],
                properties=properties,
                references=references,
            )

            success_count += 1

        except WeaviateBaseError as exc:
            logger.warning(
                "Failed to upsert IGDB entity (UUID=%s): %s",
                obj.get("uuid"),
                exc,
            )

        except Exception as exc:
            logger.warning(
                "Unexpected error for IGDB entity (UUID=%s): %s",
                obj.get("uuid"),
                exc,
            )

    return success_count


# ------------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upsert IGDB relational metadata (Weaviate v4, zero vectors)"
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

    try:
        client = weaviate.connect_to_local()
    except Exception as exc:
        raise SystemExit(f"❌ Failed to connect to Weaviate: {exc}")

    try:
        count = upsert_igdb_context(
            client=client,
            game_title=args.game,
            game_uuid=args.uuid,
        )
        print(f"✅ Upserted {count} IGDB entities")
    finally:
        client.close()
