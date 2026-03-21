from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, Optional, List

import modal
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from fastembed import SparseTextEmbedding

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Modal class lookup
# ------------------------------------------------------------------------------
E5Embedder = modal.Cls.from_name(
    "editorial-embedding-service",
    "E5Embedder",
)

# BM25 sparse encoder (lightweight, CPU-only)
bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client from environment variables."""
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    return QdrantClient(url=url, api_key=api_key or None)


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def validate_chunk(chunk_uuid: str, properties: Dict) -> bool:
    content = properties.get("content")
    if not isinstance(content, str) or not content.strip():
        return False

    game_uuid = properties.get("game_uuid")
    if not game_uuid:
        # Also check legacy beacon format for backwards compat
        game_beacon = properties.get("game", {}).get("beacon")
        if not game_beacon:
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
    payloads: List[Dict],
    batch_size: int = 64,
) -> None:
    """
    Stage 5:
    - Embed editorial chunks via Modal (E5 on T4) + BM25 sparse
    - Upsert dual vectors into Qdrant
    """

    # --------------------------------------------------
    # 1. Validate payloads (in-memory — no file I/O)
    # --------------------------------------------------
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
    # 2. Dense embedding via Modal (E5)
    # --------------------------------------------------
    embedder = E5Embedder()

    dense_vectors: List[List[float]] = []
    total_batches = (len(texts) - 1) // batch_size + 1

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        logger.info(
            "Embedding batch %d / %d",
            (i // batch_size) + 1,
            total_batches,
        )

        batch_vectors = embedder.embed_texts.remote(batch_texts)
        dense_vectors.extend(batch_vectors)

    if len(dense_vectors) != len(valid_objects):
        raise RuntimeError(
            f"Embedding mismatch: {len(dense_vectors)} vectors for {len(valid_objects)} chunks"
        )

    logger.info("Generated dense embeddings for %d chunks", len(dense_vectors))

    # --------------------------------------------------
    # 3. BM25 sparse embedding (local, CPU)
    # --------------------------------------------------
    sparse_embeddings = list(bm25_encoder.passage_embed(texts))

    if len(sparse_embeddings) != len(valid_objects):
        raise RuntimeError(
            f"Sparse embedding mismatch: {len(sparse_embeddings)} for {len(valid_objects)} chunks"
        )

    logger.info("Generated BM25 sparse embeddings for %d chunks", len(sparse_embeddings))

    # --------------------------------------------------
    # 4. Upsert into Qdrant (dual vectors)
    # --------------------------------------------------
    client = _get_qdrant_client()

    points: List[PointStruct] = []
    for obj, dense_vec, sparse_emb in zip(valid_objects, dense_vectors, sparse_embeddings):
        props = obj["properties"]

        # Extract game_uuid from beacon or direct field
        game_uuid = props.get("game_uuid")
        if not game_uuid:
            game_beacon = props.get("game", {}).get("beacon", "")
            game_uuid = game_beacon.rstrip("/").split("/")[-1] if game_beacon else None

        parent_uuid = props.get("parent_editorial_uuid")
        if not parent_uuid:
            parent_beacon = props.get("parent_editorial", {}).get("beacon", "")
            parent_uuid = parent_beacon.rstrip("/").split("/")[-1] if parent_beacon else None

        points.append(
            PointStruct(
                id=obj["uuid"],
                vector={
                    "dense": dense_vec,
                    "bm25": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload={
                    "content": props["content"],
                    "content_hash": props.get("content_hash"),
                    "source_title": props.get("source_title"),
                    "source": props.get("source"),
                    "content_type": props.get("content_type"),
                    "chunk_index": props.get("chunk_index"),
                    "game_uuid": game_uuid,
                    "parent_editorial_uuid": parent_uuid,
                },
            )
        )

    # Batch upsert
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name="EditorialChunk",
            points=batch,
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
        description="Stage 5: Embed + Upsert Editorial Chunks (Modal + Qdrant)"
    )
    parser.add_argument(
        "--file",
        required=False,
        help="Prepared editorial chunk payload JSON file (legacy)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size",
    )

    args = parser.parse_args()

    # Legacy file-based path for CLI testing
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            payloads = json.load(f)
        upsert_chunk_batch(
            payloads=payloads,
            batch_size=args.batch_size,
        )
    else:
        print("Usage: provide --file for legacy JSON payloads, or call upsert_chunk_batch() in-memory")
