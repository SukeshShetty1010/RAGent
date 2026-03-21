#!/usr/bin/env python3
"""
Master Upsert Orchestrator (FULLY OBSERVABLE)

Stages:
1. Canonical Game Anchor
2. Platform Specs
3. IGDB Metadata
4. GameSpot Editorial Container
5. Editorial Chunking + Embedding + Insert (IN-MEMORY — no JSON file I/O)
"""

from __future__ import annotations

import sys
import os
import re
import json
import logging
import argparse

from qdrant_client import QdrantClient

from utils.observability import ProfileBlock, MetricsRegistry

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
    """
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client from environment variables."""
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    return QdrantClient(url=url, api_key=api_key or None)


# -------------------------------------------------------------------
# MAIN ORCHESTRATOR
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Qdrant ingestion pipeline (Stages 1–5)."
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Canonical game name (e.g., 'Far Cry 5')",
    )
    args = parser.parse_args()

    client: QdrantClient | None = None

    with ProfileBlock("INGEST_TOTAL"):
        try:
            # ------------------------------------------------------------
            # Client lifecycle
            # ------------------------------------------------------------
            logger.info("Connecting to Qdrant...")
            client = _get_qdrant_client()

            # ------------------------------------------------------------
            # Stage 1: Canonical Game Anchor
            # ------------------------------------------------------------
            with ProfileBlock("Stage1_CanonicalGame"):
                logger.info("Starting Stage 1: Canonical Game Anchor...")
                game_uuid = upsert_game_anchor(client, args.game)

            if not game_uuid or not isinstance(game_uuid, str):
                raise RuntimeError("Invalid UUID returned from upsert_game_anchor")

            logger.info(f"✅ Stage 1 Complete. Anchor UUID: {game_uuid}")

            # ------------------------------------------------------------
            # Stage 2: Platform Specs
            # ------------------------------------------------------------
            with ProfileBlock("Stage2_PlatformSpecs"):
                logger.info("Starting Stage 2: Platform Specs...")
                spec_count = upsert_platform_specs(
                    client=client,
                    game_name=args.game,
                    game_uuid=game_uuid,
                )

            MetricsRegistry.get().observe(
                "platform_specs_upserted", spec_count
            )

            logger.info(
                f"✅ Stage 2 Complete. Upserted {spec_count} Platform Specs."
            )

            # ------------------------------------------------------------
            # Stage 3: IGDB Metadata
            # ------------------------------------------------------------
            with ProfileBlock("Stage3_IGDBMetadata"):
                logger.info("Starting Stage 3: IGDB Metadata...")
                igdb_count = upsert_igdb_context(
                    client=client,
                    game_title=args.game,
                    game_uuid=game_uuid,
                )

            MetricsRegistry.get().observe(
                "igdb_entities_upserted", igdb_count
            )

            logger.info(
                f"✅ Stage 3 Complete. Upserted {igdb_count} IGDB entities."
            )

            # ------------------------------------------------------------
            # Stage 4: GameSpot Editorial Container (FAIL-SOFT)
            # ------------------------------------------------------------
            with ProfileBlock("Stage4_GameSpotContainer"):
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
            # Stage 5: Editorial Chunking + Embedding + Insert
            #           (IN-MEMORY — no JSON file I/O)
            # ------------------------------------------------------------
            logger.info("Starting Stage 5: Editorial Chunking & Embedding...")

            try:
                with ProfileBlock("Stage5_ChunkGeneration"):
                    chunks = generate_chunk_payloads(args.game, game_uuid)

                if not chunks:
                    logger.warning(
                        "⚠️ No editorial chunks generated. Skipping Stage 5."
                    )
                else:
                    MetricsRegistry.get().observe(
                        "editorial_chunks_generated", len(chunks)
                    )

                    logger.info(
                        f"Generated {len(chunks)} editorial chunks (in-memory)"
                    )

                    with ProfileBlock("Stage5_VectorInsert"):
                        upsert_chunk_batch(chunks)

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
                logger.info("Closing Qdrant connection...")
                client.close()

            # ------------------------------------------------------------
            # Final ingestion observability report
            # ------------------------------------------------------------
            print("\n--- INGEST OBSERVABILITY REPORT ---")
            print(
                json.dumps(
                    MetricsRegistry.get().generate_report(),
                    indent=2
                )
            )


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
