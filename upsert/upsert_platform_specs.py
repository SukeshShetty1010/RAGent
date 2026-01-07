from __future__ import annotations

import argparse
import logging
from typing import Dict, Optional

import weaviate
from weaviate.exceptions import WeaviateBaseError

from data.rawg_data import fetch_rawg_game_data
from pre_process.cleaner import RAWGCleaner
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
def upsert_platform_specs(
    client: weaviate.WeaviateClient,
    game_name: str,
    game_uuid: str,
) -> int:
    """
    Fetch RAWG data, extract platform specs, and upsert PlatformSpec
    objects into Weaviate (v4).

    Design:
    - Deterministic UUIDs
    - Idempotent inserts
    - Soft-failure per object
    """

    # --------------------------------------------------
    # 1. Fetch RAWG
    # --------------------------------------------------
    try:
        rawg_raw = fetch_rawg_game_data(game_name)
    except Exception as exc:
        logger.warning(
            "RAWG fetch failed for '%s': %s. Skipping PlatformSpec ingest.",
            game_name,
            exc,
        )
        return 0

    if not rawg_raw:
        logger.warning("No RAWG data returned for '%s'.", game_name)
        return 0

    # --------------------------------------------------
    # 2. Clean RAWG
    # --------------------------------------------------
    cleaner = RAWGCleaner()
    try:
        cleaned = cleaner.clean(rawg_raw)
    except Exception as exc:
        logger.warning(
            "RAWG cleaning failed for '%s': %s. Skipping PlatformSpec ingest.",
            game_name,
            exc,
        )
        return 0

    if not cleaned:
        logger.info("RAWG cleaner produced no usable data for '%s'.", game_name)
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
    # 4. Upsert (v4 collections, idempotent)
    # --------------------------------------------------
    collection = client.collections.get("PlatformSpec")

    success_count = 0

    for obj in payloads:
        platform_name = obj["properties"].get("platform_name", "UNKNOWN")

        try:
            # ---- Idempotency gate ----
            if collection.data.exists(uuid=obj["uuid"]):
                logger.info(
                    "PlatformSpec already exists for '%s' (%s). Skipping.",
                    game_name,
                    platform_name,
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
                "Failed to upsert PlatformSpec for '%s' (%s): %s",
                game_name,
                platform_name,
                exc,
            )

        except Exception as exc:
            logger.warning(
                "Unexpected error for PlatformSpec '%s' (%s): %s",
                game_name,
                platform_name,
                exc,
            )

    return success_count


# ------------------------------------------------------------------
# CLI Entrypoint (Weaviate v4)
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upsert PlatformSpec objects (Weaviate v4, filter-only)"
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

    try:
        client = weaviate.connect_to_local()
    except Exception as exc:
        raise SystemExit(f"❌ Failed to connect to Weaviate: {exc}")

    try:
        count = upsert_platform_specs(
            client=client,
            game_name=args.game,
            game_uuid=args.uuid,
        )
        print(f"✅ Upserted {count} PlatformSpecs for {args.game}")
    finally:
        client.close()
