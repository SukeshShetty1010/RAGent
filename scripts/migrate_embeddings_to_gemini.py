#!/usr/bin/env python3
"""
scripts/migrate_embeddings_to_gemini.py

One-time, resumable migration of the EditorialChunk collection's "dense"
named vector from Modal-hosted E5-base-v2 to Gemini gemini-embedding-001
(768-dim, L2-normalized). Only the dense vector is touched — payloads,
point IDs, and the "bm25" sparse vector are left exactly as they are. No
re-ingest from source APIs, no collection recreation, no data loss.

Query vectors and stored vectors must live in the same embedding space,
so this must complete before the new Gemini-embedding query path (see
retriever/rag_retriever.py) serves a live request against this corpus.

Usage:
    python -m scripts.migrate_embeddings_to_gemini --dry-run
    python -m scripts.migrate_embeddings_to_gemini
    python -m scripts.migrate_embeddings_to_gemini --resume   # after a Ctrl-C or 429 wall
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointVectors

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from llm.gemini_client import embed_texts

COLLECTION = "EditorialChunk"
CHECKPOINT_FILE = os.path.join(
    os.path.dirname(__file__), "..", ".migration_checkpoint_gemini_embed.json"
)
SCROLL_BATCH = 50
EMBED_RETRIES = 4
# Gemini's free-tier embedding RPM is far below the batchEmbedContents
# request cap (confirmed live: SCROLL_BATCH=50 back-to-back triggers
# sustained 429s) -- pace successive batches instead of relying on
# per-batch backoff alone.
INTER_BATCH_SLEEP_SECONDS = 6.0


def _load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return set(json.load(f))


def _save_checkpoint(done_ids: set[str]) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(done_ids), f)


def _get_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    return QdrantClient(url=url, api_key=api_key or None, timeout=60)


def _embed_with_retry(texts: list[str]) -> list[list[float]]:
    delay = 2.0
    for attempt in range(EMBED_RETRIES):
        try:
            return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
        except Exception as exc:
            if attempt == EMBED_RETRIES - 1:
                raise
            print(f"  embed batch failed ({exc}); retrying in {delay}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def migrate(dry_run: bool, resume: bool) -> None:
    client = _get_client()
    done_ids = _load_checkpoint() if resume else set()
    if resume and done_ids:
        print(f"Resuming — {len(done_ids)} points already migrated.")

    total = client.count(COLLECTION, exact=True).count
    print(f"{COLLECTION}: {total} total points.")

    if dry_run:
        remaining = total - len(done_ids)
        # batchEmbedContents caps at 100 requests/call; report expected call volume.
        print(f"Dry run: {remaining} points would be re-embedded.")
        print(f"Estimated Gemini embedContent calls: {remaining} (1 per chunk).")
        return

    migrated = 0
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["content"],
            with_vectors=False,
        )
        if not records:
            break

        pending = [r for r in records if str(r.id) not in done_ids]
        if pending:
            texts = [r.payload.get("content") or "" for r in pending]
            vectors = _embed_with_retry(texts)

            points = [
                PointVectors(id=r.id, vector={"dense": vec})
                for r, vec in zip(pending, vectors)
            ]
            client.update_vectors(collection_name=COLLECTION, points=points)

            for r in pending:
                done_ids.add(str(r.id))
            migrated += len(pending)
            _save_checkpoint(done_ids)
            print(f"  migrated {migrated} points so far...")
            time.sleep(INTER_BATCH_SLEEP_SECONDS)

        if offset is None:
            break

    client.close()

    from utils.usage_counter import UsageCounter
    UsageCounter.get().flush()

    print(f"\nDone. {migrated} points re-embedded this run, {len(done_ids)} total complete.")
    if len(done_ids) >= total:
        print("All points migrated — safe to remove the checkpoint file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report point count and estimated call volume only")
    parser.add_argument("--resume", action="store_true", help="Skip points already recorded in the checkpoint file")
    args = parser.parse_args()

    migrate(dry_run=args.dry_run, resume=args.resume)
