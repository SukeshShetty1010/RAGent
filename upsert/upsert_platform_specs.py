from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from data.rawg_data import fetch_rawg_game_data, rawg_available
from pre_process.cleaner import RAWGCleaner
from ingest.platform_sources import build_platform_cleaned_data
from ingest.platformspec_ingest import generate_platform_payloads


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
def upsert_platform_specs(
    client: QdrantClient,
    game_name: str,
    game_uuid: str,
) -> int:
    """
    Fetch RAWG data, extract platform specs, and upsert PlatformSpec
    objects into Qdrant.

    Design:
    - Deterministic UUIDs
    - Idempotent inserts
    - Soft-failure per object
    """

    # --------------------------------------------------
    # 1. Fetch platform data — RAWG when its circuit breaker is closed,
    #    else fall through to the provider-independent IGDB+Steam
    #    substitute (ingest/platform_sources.py).
    # --------------------------------------------------
    cleaned: Optional[Dict] = None

    if rawg_available():
        try:
            rawg_raw = fetch_rawg_game_data(game_name)
        except Exception as exc:
            logger.warning(
                "RAWG fetch failed for '%s': %s. Falling back to IGDB+Steam.",
                game_name,
                exc,
            )
            rawg_raw = None

        if rawg_raw:
            cleaner = RAWGCleaner()
            try:
                cleaned = cleaner.clean(rawg_raw)
            except Exception as exc:
                logger.warning(
                    "RAWG cleaning failed for '%s': %s. Falling back to IGDB+Steam.",
                    game_name,
                    exc,
                )
                cleaned = None
    else:
        logger.info(
            "RAWG unavailable (disabled or circuit open) — using IGDB+Steam for '%s'.",
            game_name,
        )

    if not cleaned:
        try:
            cleaned = build_platform_cleaned_data(game_name)
        except Exception as exc:
            logger.warning(
                "IGDB+Steam platform fallback failed for '%s': %s.",
                game_name,
                exc,
            )
            cleaned = None

    if not cleaned:
        logger.info("No platform data available for '%s' from any source.", game_name)
        return 0

    # --------------------------------------------------
    # 3. Generate PlatformSpec payloads
    # --------------------------------------------------
    payloads = generate_platform_payloads(
        cleaned_data=cleaned,
        game_uuid=game_uuid,
    )

    if not payloads:
        logger.info("No platform specs found for '%s'.", game_name)
        return 0

    # --------------------------------------------------
    # 4. Upsert (Qdrant, idempotent)
    # --------------------------------------------------
    success_count = 0

    for obj in payloads:
        platform_name = obj["properties"].get("platform_name", "UNKNOWN")

        try:
            # ---- Idempotency gate ----
            existing = client.retrieve(
                collection_name="PlatformSpec",
                ids=[obj["uuid"]],
            )

            if existing:
                logger.info(
                    "PlatformSpec already exists for '%s' (%s). Skipping.",
                    game_name,
                    platform_name,
                )
                continue

            # ---- Prepare payload (strip beacon refs) ----
            properties: Dict = dict(obj["properties"])
            game_ref = properties.pop("game", None)

            # Store game_uuid as a payload field instead of a beacon reference
            properties["game_uuid"] = game_uuid

            # ---- Insert ----
            client.upsert(
                collection_name="PlatformSpec",
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
                "Failed to upsert PlatformSpec for '%s' (%s): %s",
                game_name,
                platform_name,
                exc,
            )

    return success_count


# ------------------------------------------------------------------
# CLI Entrypoint (Qdrant)
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upsert PlatformSpec objects (Qdrant, filter-only)"
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Game name (RAWG lookup)",
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
        count = upsert_platform_specs(
            client=client,
            game_name=args.game,
            game_uuid=args.uuid,
        )
        print(f"✅ Upserted {count} PlatformSpecs for {args.game}")
    finally:
        client.close()
