from __future__ import annotations

import argparse
import logging

import weaviate
from weaviate.collections.classes.batch import BatchObjectReturn

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
# Core Upsert Function
# ------------------------------------------------------------------
def upsert_igdb_context(
    client: weaviate.WeaviateClient,
    game_title: str,
    game_uuid: str,
) -> int:
    """
    Fetch and upsert IGDB relational metadata using Weaviate v4.

    Soft-failure by design:
    - Fetch failures are logged and skipped
    - Batch insert errors are logged but do not raise
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
    # 2. Get Collection (v4)
    # --------------------------------------------------
    igdb_col = client.collections.get("IGDB_Game")

    # --------------------------------------------------
    # 3. Batch Insert (v4 Dynamic Batch)
    # --------------------------------------------------
    with igdb_col.batch.dynamic() as batch:
        for obj in payloads:
            batch.add_object(
                properties=obj["properties"],
                uuid=obj["uuid"],
            )

    # --------------------------------------------------
    # 4. Inspect Batch Results (SOFT FAILURE)
    # --------------------------------------------------
    failed = batch.failed_objects or []

    for failure in failed:
        logger.warning(
            "IGDB upsert failed (UUID=%s): %s",
            failure.object_.uuid if isinstance(failure, BatchObjectReturn) else "UNKNOWN",
            failure.error,
        )

    success_count = len(payloads) - len(failed)
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
