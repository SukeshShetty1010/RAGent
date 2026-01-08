from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, Optional, List

import modal
import weaviate

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Modal class lookup (CORRECT + FUTURE-PROOF)
# ------------------------------------------------------------------------------
E5Embedder = modal.Cls.from_name(
    "editorial-embedding-service",
    "E5Embedder",
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def uuid_from_beacon(beacon: Optional[str]) -> Optional[str]:
    if not beacon or not isinstance(beacon, str):
        return None
    return beacon.rstrip("/").split("/")[-1]


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def validate_chunk(chunk_uuid: str, properties: Dict) -> bool:
    content = properties.get("content")
    if not isinstance(content, str) or not content.strip():
        return False

    game_beacon = properties.get("game", {}).get("beacon")
    if not uuid_from_beacon(game_beacon):
        return False

    if properties.get("source") != "gamespot":
        return False

    if properties.get("content_type") not in {"review", "article"}:
        return False

    return True


# ------------------------------------------------------------------------------
# Stage 5 Orchestration
# ------------------------------------------------------------------------------

def upsert_chunk_batch(
    file_path: str,
    batch_size: int = 64,
) -> None:
    """
    Stage 5:
    - Embed editorial chunks via Modal (E5 on T4)
    - Upsert vectors into Weaviate v4
    """

    # --------------------------------------------------
    # 1. Load payloads
    # --------------------------------------------------
    with open(file_path, "r", encoding="utf-8") as f:
        payloads = json.load(f)

    valid_objects: List[Dict] = []
    texts: List[str] = []

    for obj in payloads:
        chunk_uuid = obj.get("uuid")
        props = obj.get("properties", {})

        if chunk_uuid and validate_chunk(chunk_uuid, props):
            valid_objects.append(obj)
            texts.append(props["content"])

    if not valid_objects:
        logger.warning("No valid editorial chunks to process.")
        return

    logger.info("Validated %d editorial chunks", len(valid_objects))

    # --------------------------------------------------
    # 2. Embed via Modal (STATEFUL INSTANCE)
    # --------------------------------------------------
    embedder = E5Embedder()  # 🔑 THIS triggers __enter__()

    vectors: List[List[float]] = []
    total_batches = (len(texts) - 1) // batch_size + 1

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        logger.info(
            "Embedding batch %d / %d",
            (i // batch_size) + 1,
            total_batches,
        )

        batch_vectors = embedder.embed_texts.remote(batch_texts)
        vectors.extend(batch_vectors)

    if len(vectors) != len(valid_objects):
        raise RuntimeError(
            f"Embedding mismatch: {len(vectors)} vectors for {len(valid_objects)} chunks"
        )

    logger.info("Generated embeddings for %d chunks", len(vectors))

    # --------------------------------------------------
    # 3. Upsert into Weaviate
    # --------------------------------------------------
    client = weaviate.connect_to_local()
    collection = client.collections.get("EditorialChunk")

    with collection.batch.dynamic() as batch:
        for obj, vector in zip(valid_objects, vectors):
            props = obj["properties"]

            clean_props = dict(props)
            clean_props.pop("game", None)
            clean_props.pop("parent_editorial", None)

            references: Dict[str, str] = {}

            game_uuid = uuid_from_beacon(props["game"]["beacon"])
            if game_uuid:
                references["game"] = game_uuid

            parent_beacon = props.get("parent_editorial", {}).get("beacon")
            parent_uuid = uuid_from_beacon(parent_beacon)
            if parent_uuid:
                references["parent_editorial"] = parent_uuid

            batch.add_object(
                uuid=obj["uuid"],
                properties=clean_props,
                references=references,
                vector=vector,
            )

    client.close()

    logger.info(
        "✅ Stage 5 complete — %d editorial chunks embedded & upserted",
        len(valid_objects),
    )


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 5: Embed + Upsert Editorial Chunks (Modal + Weaviate)"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Prepared editorial chunk payload JSON file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size",
    )

    args = parser.parse_args()
    upsert_chunk_batch(
        file_path=args.file,
        batch_size=args.batch_size,
    )
