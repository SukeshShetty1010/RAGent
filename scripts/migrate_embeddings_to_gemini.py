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

    # Self-detecting repair for a checkpoint-less partial migration: finds
    # any leftover E5 vectors by clustering on cosine-to-centroid distance
    # (E5 and Gemini vectors are both 768-dim and both L2-normalized, so
    # neither shape nor norm tells them apart -- direction does) and
    # re-embeds only that cluster. No checkpoint file needed; safe to
    # re-run, and a no-op on an already-uniform corpus.
    python -m scripts.migrate_embeddings_to_gemini --repair --dry-run
    python -m scripts.migrate_embeddings_to_gemini --repair
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
# The measured split (2026-08-21, EditorialChunk) had a ~0.65-wide empty
# gap between the E5 and Gemini clusters (E5: -0.066..0.027, Gemini:
# 0.679..0.865 cosine-to-centroid). 0.2 is a wide margin under that --
# a corpus mid-migration or already uniform won't produce a gap this
# large, so falling short of it means "nothing to repair," not "guess."
REPAIR_MIN_GAP = 0.2
REPAIR_SAMPLE_SIZE = 3


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


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _centroid(unit_vectors: list[list[float]]) -> list[float]:
    dim = len(unit_vectors[0])
    sums = [0.0] * dim
    for v in unit_vectors:
        for i, x in enumerate(v):
            sums[i] += x
    n = len(unit_vectors)
    return _normalize([s / n for s in sums])


def find_gap_split(vectors: list[list[float]], min_gap: float = REPAIR_MIN_GAP):
    """
    Score every vector by cosine similarity to the corpus centroid, then
    split at the single largest gap in the sorted score distribution.

    Returns (cluster_a_idx, cluster_b_idx) -- two lists of indices into
    `vectors`, low-scoring and high-scoring respectively -- or None if no
    gap at least `min_gap` wide exists. A corpus that is already uniform
    (fully migrated, or not migrated at all) has no such gap, and picking
    a split anyway would silently re-embed the wrong half.
    """
    if len(vectors) < 2:
        return None

    unit_vectors = [_normalize(v) for v in vectors]
    centroid = _centroid(unit_vectors)
    scores = [_dot(v, centroid) for v in unit_vectors]

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    sorted_scores = [scores[i] for i in order]
    gaps = [sorted_scores[i + 1] - sorted_scores[i] for i in range(len(sorted_scores) - 1)]
    if not gaps:
        return None

    gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    if gaps[gap_idx] < min_gap:
        return None

    low_idx = order[: gap_idx + 1]
    high_idx = order[gap_idx + 1 :]
    return low_idx, high_idx


def identify_migrated_cluster(
    cluster_a: list[int],
    cluster_b: list[int],
    contents: list[str],
    vectors: list[list[float]],
    embed_fn=None,
    sample_size: int = REPAIR_SAMPLE_SIZE,
):
    """
    Decide which of the two clusters is already-migrated (Gemini) by
    re-embedding a small sample from each and comparing the stored vector
    against a fresh embedding of the same content -- the migrated cluster
    scores ~1.0, the unmigrated one does not.

    Deliberately does not use cluster size as a signal: when the
    unmigrated points are the majority (the common case for a stalled
    migration), a majority-vote heuristic would pick the wrong cluster.

    Returns (migrated_idx, unmigrated_idx).
    """
    if embed_fn is None:
        embed_fn = lambda texts: _embed_with_retry(texts)  # noqa: E731

    def self_cos(idx_list: list[int]) -> float:
        sample = idx_list[:sample_size]
        texts = [contents[i] for i in sample]
        fresh_vectors = embed_fn(texts)
        cos_vals = []
        for i, fresh in zip(sample, fresh_vectors):
            stored = _normalize(vectors[i])
            fresh_unit = _normalize(fresh)
            cos_vals.append(_dot(stored, fresh_unit))
        return sum(cos_vals) / len(cos_vals)

    cos_a = self_cos(cluster_a)
    cos_b = self_cos(cluster_b)

    if cos_a >= cos_b:
        return cluster_a, cluster_b
    return cluster_b, cluster_a


def _fetch_all_dense_vectors(client: QdrantClient):
    """Returns (ids, vectors, contents) for every point in COLLECTION."""
    ids: list = []
    vectors: list[list[float]] = []
    contents: list[str] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["content"],
            with_vectors=True,
        )
        if not records:
            break
        for r in records:
            ids.append(r.id)
            vectors.append(r.vector["dense"])
            contents.append(r.payload.get("content") or "")
        if offset is None:
            break
    return ids, vectors, contents


def repair(dry_run: bool) -> None:
    client = _get_client()
    total = client.count(COLLECTION, exact=True).count
    print(f"{COLLECTION}: {total} total points. Scanning for a leftover embedding-space split...")

    ids, vectors, contents = _fetch_all_dense_vectors(client)
    split = find_gap_split(vectors)

    if split is None:
        print("No wide gap in the cosine-to-centroid distribution — corpus looks uniform. Nothing to repair.")
        client.close()
        return

    cluster_a, cluster_b = split
    print(f"Found a split: cluster A={len(cluster_a)} points, cluster B={len(cluster_b)} points.")
    print(f"Verifying direction by re-embedding {REPAIR_SAMPLE_SIZE} samples from each cluster...")

    migrated_idx, unmigrated_idx = identify_migrated_cluster(cluster_a, cluster_b, contents, vectors)
    print(f"Identified {len(migrated_idx)} already-migrated points and {len(unmigrated_idx)} unmigrated points.")

    if dry_run:
        print(f"Dry run: {len(unmigrated_idx)} points would be re-embedded.")
        client.close()
        return

    unmigrated_ids = [ids[i] for i in unmigrated_idx]
    unmigrated_contents = [contents[i] for i in unmigrated_idx]

    repaired = 0
    for start in range(0, len(unmigrated_ids), SCROLL_BATCH):
        batch_ids = unmigrated_ids[start : start + SCROLL_BATCH]
        batch_texts = unmigrated_contents[start : start + SCROLL_BATCH]
        batch_vectors = _embed_with_retry(batch_texts)

        points = [
            PointVectors(id=pid, vector={"dense": vec})
            for pid, vec in zip(batch_ids, batch_vectors)
        ]
        client.update_vectors(collection_name=COLLECTION, points=points)

        repaired += len(batch_ids)
        print(f"  repaired {repaired}/{len(unmigrated_ids)} points so far...")
        if start + SCROLL_BATCH < len(unmigrated_ids):
            time.sleep(INTER_BATCH_SLEEP_SECONDS)

    client.close()

    from utils.usage_counter import UsageCounter
    UsageCounter.get().flush()

    print(f"\nRepair done. {repaired} points re-embedded.")
    print("Re-run --repair --dry-run to confirm 0 points remain unmigrated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report point count and estimated call volume only")
    parser.add_argument("--resume", action="store_true", help="Skip points already recorded in the checkpoint file")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Self-detecting repair for a checkpoint-less partial migration (see module docstring)",
    )
    args = parser.parse_args()

    if args.repair:
        repair(dry_run=args.dry_run)
    else:
        migrate(dry_run=args.dry_run, resume=args.resume)
