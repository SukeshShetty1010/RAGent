from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, Optional, List

import weaviate

# Modal embedder (class-based, warm containers)
from llm.modal_embed import E5Embedder


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helper: Extract UUID from Beacon
# ------------------------------------------------------------------
def uuid_from_beacon(beacon: Optional[str]) -> Optional[str]:
    """
    Convert:
      weaviate://localhost/Game/<uuid>
    → <uuid>
    """
    if not beacon or not isinstance(beacon, str):
        return None
    try:
        return beacon.rstrip("/").split("/")[-1]
    except Exception:
        return None


# ------------------------------------------------------------------
# Validator (Structural + Semantic Gate)
# ------------------------------------------------------------------
def validate_chunk(chunk_uuid: str, properties: Dict) -> bool:
    """
    Validate editorial chunk BEFORE embedding & upsert.
    """

    # ---- Required text ----
    content = properties.get("content")
    if not isinstance(content, str) or not content.strip():
        logger.warning("Skipping %s: Missing or empty content", chunk_uuid)
        return False

    # ---- Canonical Game reference (No Orphan Rule) ----
    game_beacon = properties.get("game", {}).get("beacon")
    if not uuid_from_beacon(game_beacon):
        logger.warning("Skipping %s: Missing canonical Game reference", chunk_uuid)
        return False

    # ---- Source integrity ----
    if properties.get("source") != "gamespot":
        logger.warning("Skipping %s: Invalid source", chunk_uuid)
        return False

    # ---- Content type gate ----
    if properties.get("content_type") not in {"review", "article"}:
        logger.warning("Skipping %s: Invalid content_type", chunk_uuid)
        return False

    return True


# ------------------------------------------------------------------
# Core Orchestration
# ------------------------------------------------------------------
def upsert_chunk_batch(file_path: str) -> None:
    """
    Validate → Embed (Modal e5-base-v2) → Upsert Editorial Chunks (Weaviate v4).

    - Client-side embeddings
    - Explicit vectors passed to Weaviate
    - No OpenAI
    """

    # --------------------------------------------------
    # 1. Load payloads
    # --------------------------------------------------
    with open(file_path, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    if not isinstance(payloads, list):
        raise ValueError("Payload file must contain a list")

    total = len(payloads)
    upserted = 0
    skipped = 0
    db_errors = 0

    # --------------------------------------------------
    # 2. Validate & prepare embedding batch
    # --------------------------------------------------
    contents: List[str] = []
    prepared_objects: List[Dict] = []

    for obj in payloads:
        chunk_uuid = obj.get("uuid")
        properties = obj.get("properties", {})

        if not chunk_uuid:
            logger.warning("Skipping object with missing UUID")
            skipped += 1
            continue

        if not validate_chunk(chunk_uuid, properties):
            skipped += 1
            continue

        contents.append(properties["content"])
        prepared_objects.append(obj)

    if not prepared_objects:
        logger.warning("No valid editorial chunks to upsert.")
        return

    # --------------------------------------------------
    # 3. Generate embeddings via Modal (batched)
    # --------------------------------------------------
    logger.info(
        "Generating embeddings for %d editorial chunks using e5-base-v2 (Modal)...",
        len(contents),
    )

    embedder = E5Embedder()

    vectors = embedder.embed_text.remote(
        contents,
        mode="passage",  # IMPORTANT: E5 requires passage prefix for corpus
    )

    if len(vectors) != len(prepared_objects):
        raise RuntimeError("Embedding count mismatch — aborting upsert")

    # --------------------------------------------------
    # 4. Connect to Weaviate (v4)
    # --------------------------------------------------
    client = weaviate.connect_to_local()
    collection = client.collections.get("EditorialChunk")

    # --------------------------------------------------
    # 5. Batch Insert (explicit vectors)
    # --------------------------------------------------
    with collection.batch.dynamic() as batch:
        for obj, vector in zip(prepared_objects, vectors):
            chunk_uuid = obj["uuid"]
            properties = obj["properties"]

            # ---- Separate properties and references ----
            clean_properties = dict(properties)
            clean_properties.pop("game", None)
            clean_properties.pop("parent_editorial", None)

            references: Dict[str, str] = {}

            # Canonical Game reference
            game_uuid = uuid_from_beacon(
                properties.get("game", {}).get("beacon")
            )
            if game_uuid:
                references["game"] = game_uuid

            # Parent GameSpot editorial reference
            parent_uuid = uuid_from_beacon(
                properties.get("parent_editorial", {}).get("beacon")
            )
            if parent_uuid:
                references["parent_editorial"] = parent_uuid

            # ---- Add object with explicit vector ----
            batch.add_object(
                uuid=chunk_uuid,
                properties=clean_properties,
                references=references,
                vector=vector,
            )
            upserted += 1

    # --------------------------------------------------
    # 6. Batch Error Handling
    # --------------------------------------------------
    failed = collection.batch.failed_objects or []
    for failure in failed:
        logger.error(
            "DB error for chunk %s: %s",
            getattr(failure.object_, "uuid", "UNKNOWN"),
            failure.error,
        )
        db_errors += 1

    client.close()

    # --------------------------------------------------
    # 7. Summary
    # --------------------------------------------------
    print(
        f"Total Loaded: {total} | "
        f"Upserted: {upserted} | "
        f"Skipped: {skipped} | "
        f"DB Errors: {db_errors}"
    )


# ------------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upsert Editorial Chunks (Modal e5-base-v2 embeddings)"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to prepared editorial chunk JSON file",
    )

    args = parser.parse_args()
    upsert_chunk_batch(args.file)
