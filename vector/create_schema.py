#!/usr/bin/env python3
"""
Qdrant Collection Schema Creator

Creates all required collections in Qdrant:
- EditorialChunk: dual vectors (dense E5-768 + BM25 sparse with IDF)
- Game, PlatformSpec, IGDB_Game, GameSpot_Game, EditorialSource: metadata-only (1-dim dummy)
"""

from __future__ import annotations

import os
import sys

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Windows consoles/redirected-output pipes often default to cp1252, which
# can't encode the emoji status markers below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# Dense vector size (must match E5-base-v2 embedding service)
E5_VECTOR_SIZE = 768


# -------------------------------------------------------------------
# SCHEMA DEFINITIONS
# -------------------------------------------------------------------

def create_all_collections(client: QdrantClient) -> None:
    """
    Create or recreate all Qdrant collections for RAGent.

    EditorialChunk:
        - "dense": E5-base-v2 (768-dim, cosine)
        - "bm25":  BM25 sparse with server-side IDF scoring

    Game, PlatformSpec, IGDB_Game, GameSpot_Game:
        - Metadata-only collections (1-dim dummy vector)
        - Used for payload storage and filtering only
    """

    existing = {c.name for c in client.get_collections().collections}

    # --------------------------------------------------
    # EditorialChunk — hybrid search (dense + BM25)
    # --------------------------------------------------
    if "EditorialChunk" not in existing:
        client.create_collection(
            collection_name="EditorialChunk",
            vectors_config={
                "dense": models.VectorParams(
                    size=E5_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        print("✅ Created collection: EditorialChunk (dense + BM25 sparse)")
    else:
        print("⚠️  Collection already exists: EditorialChunk")

    # --------------------------------------------------
    # Metadata-only collections
    # --------------------------------------------------
    metadata_collections = [
        "Game",
        "PlatformSpec",
        "IGDB_Game",
        "GameSpot_Game",
        "EditorialSource",  # Wikipedia / Steam editorial containers
    ]

    for name in metadata_collections:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=1,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"✅ Created collection: {name} (metadata-only)")
        else:
            print(f"⚠️  Collection already exists: {name}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    print(f"Connecting to Qdrant at {QDRANT_URL}...")

    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
        )
    except Exception as exc:
        print(f"❌ Failed to connect to Qdrant: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        create_all_collections(client)
        print("\n✅ All collections ready.")
    except Exception as exc:
        print(f"❌ Schema creation failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
