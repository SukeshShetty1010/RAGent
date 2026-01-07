#!/usr/bin/env python3
"""
Master Upsert Orchestrator

Stage 1:
- Canonical Game Anchor

Stage 2:
- Platform Specs

Stage 3:
- IGDB Metadata

Future stages (stubbed):
- Editorial Chunking

Execution:
    python -m upsert.upsert_all --game "Far Cry 5"
"""

import sys
import logging
import argparse

import weaviate
from weaviate import WeaviateClient

from upsert.upsert_canonical_game import upsert_game_anchor
from upsert.upsert_platform_specs import upsert_platform_specs
from upsert.upsert_igdb_metadata import upsert_igdb_context


# -------------------------------------------------------------------
# LOGGING CONFIG
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Weaviate ingestion pipeline (Stages 1–3)."
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Canonical game name (e.g., 'Halo')",
    )
    args = parser.parse_args()

    client: WeaviateClient | None = None

    try:
        # ------------------------------------------------------------
        # Client lifecycle (single shared client)
        # ------------------------------------------------------------
        logger.info("Connecting to Weaviate...")
        client = weaviate.connect_to_local()

        # ------------------------------------------------------------
        # Stage 1: Canonical Game Anchor
        # ------------------------------------------------------------
        logger.info("Starting Stage 1: Canonical Game Anchor...")

        game_uuid = upsert_game_anchor(client, args.game)

        if not game_uuid or not isinstance(game_uuid, str):
            raise RuntimeError("Invalid UUID returned from upsert_game_anchor")

        logger.info(f"✅ Stage 1 Complete. Anchor UUID: {game_uuid}")

        # ------------------------------------------------------------
        # Stage 2: Platform Specs
        # ------------------------------------------------------------
        logger.info("Starting Stage 2: Platform Specs...")

        spec_count = upsert_platform_specs(
            client=client,
            game_name=args.game,
            game_uuid=game_uuid,
        )

        logger.info(
            f"✅ Stage 2 Complete. Upserted {spec_count} Platform Specs."
        )

        # ------------------------------------------------------------
        # Stage 3: IGDB Metadata
        # ------------------------------------------------------------
        logger.info("Starting Stage 3: IGDB Metadata...")

        igdb_count = upsert_igdb_context(
            client=client,
            game_title=args.game,
            game_uuid=game_uuid,
        )

        logger.info(
            f"✅ Stage 3 Complete. Upserted {igdb_count} IGDB entities."
        )

        # ------------------------------------------------------------
        # FUTURE STAGES (INTENTIONALLY DISABLED)
        # ------------------------------------------------------------
        # logger.info("Starting Stage 4: Editorial Chunking...")
        # upsert_editorial_chunks(client, game_uuid)

        logger.info("Pipeline execution complete (Stages 1–3).")

    except Exception as exc:
        logger.error(f"❌ Pipeline failed: {exc}")
        sys.exit(1)

    finally:
        if client is not None:
            logger.info("Closing Weaviate connection...")
            client.close()


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
