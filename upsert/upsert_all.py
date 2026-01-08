#!/usr/bin/env python3
"""
Master Upsert Orchestrator

Stage 1:
- Canonical Game Anchor

Stage 2:
- Platform Specs

Stage 3:
- IGDB Metadata

Stage 4:
- GameSpot Editorial Container

Stage 5:
- Editorial Chunking + Embedding (Modal)

Execution:
    python -m upsert.upsert_all --game "Far Cry 5"
"""

from __future__ import annotations

import sys
import os
import re
import json
import logging
import argparse

import weaviate
from weaviate import WeaviateClient

from upsert.upsert_canonical_game import upsert_game_anchor
from upsert.upsert_platform_specs import upsert_platform_specs
from upsert.upsert_igdb_metadata import upsert_igdb_context
from upsert.upsert_gamespot_chunks import upsert_gamespot_container

from embed.prepare_editorial_payloads import generate_chunk_payloads
from upsert.upsert_editorial_chunks import upsert_chunk_batch


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
# HELPERS
# -------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """
    Convert game name into filesystem-safe identifier.
    Example: "Far Cry 5" -> "far_cry_5"
    """
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


# -------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Weaviate ingestion pipeline (Stages 1–5)."
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Canonical game name (e.g., 'Far Cry 5')",
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
        # Stage 4: GameSpot Editorial Container (FAIL-SOFT)
        # ------------------------------------------------------------
        logger.info("Starting Stage 4: GameSpot Editorial Container...")

        try:
            gamespot_uuid = upsert_gamespot_container(
                client=client,
                game_name=args.game,
                game_uuid=game_uuid,
            )

            if gamespot_uuid:
                logger.info(
                    f"✅ Stage 4 Complete. GameSpot Container UUID: {gamespot_uuid}"
                )
            else:
                logger.warning(
                    "⚠️ Stage 4 Skipped (No GameSpot data found)."
                )

        except Exception as exc:
            logger.warning(
                "⚠️ Stage 4 failed but pipeline will continue: %s",
                exc,
            )

        # ------------------------------------------------------------
        # Stage 5: Editorial Chunking + Embedding (FAIL-SOFT)
        # ------------------------------------------------------------
        logger.info("Starting Stage 5: Editorial Chunking & Embedding...")

        try:
            chunks = generate_chunk_payloads(args.game, game_uuid)

            if not chunks:
                logger.warning("⚠️ No editorial chunks generated. Skipping Stage 5.")
            else:
                safe = _safe_name(args.game)
                os.makedirs("data", exist_ok=True)
                file_path = os.path.join(
                    "data", f"{safe}_editorial_chunks.json"
                )

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)

                logger.info(
                    f"Saved {len(chunks)} editorial chunks to {file_path}"
                )

                upsert_chunk_batch(file_path)

                logger.info("✅ Stage 5 Complete.")

        except Exception as exc:
            logger.warning(
                "⚠️ Stage 5 failed but pipeline will continue: %s",
                exc,
            )

        logger.info("Pipeline execution complete (Stages 1–5).")

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
