#!/usr/bin/env python3
"""
scripts/verify_corpus.py — Corpus Integrity Verifier

Nothing else in this codebase can answer "is the rebuilt corpus
actually good?" This script scans the live Qdrant cluster and reports:

  - total EditorialChunk points, and per-game_uuid chunk counts
  - games with a Game anchor but zero editorial chunks (informational —
    GameSpot genuinely has no coverage for some titles; not a violation)
  - no-orphan violations (chunking/chunk_contract.md): chunks missing
    game_uuid / parent_editorial_uuid, or whose game_uuid has no
    matching Game point
  - duplicate unified_game_id across distinct Game points (a slug
    collision in ingest/identity_resolver.py's identity scheme)

Exits non-zero when any integrity violation (orphans, duplicates) is
found — clean if the corpus is merely incomplete (zero-chunk games).

Usage:
    python -m scripts.verify_corpus
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient

SCROLL_BATCH = 256


def _get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    return QdrantClient(url=url, api_key=api_key or None)


def _scroll_all(client: QdrantClient, collection_name: str) -> List[Dict[str, Any]]:
    """Fetch every point's payload + id from a collection."""
    points: List[Dict[str, Any]] = []
    offset = None

    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            payload = dict(point.payload or {})
            payload["_id"] = point.id
            points.append(payload)

        if offset is None:
            break

    return points


def verify_corpus() -> int:
    client = _get_qdrant_client()

    try:
        games = _scroll_all(client, "Game")
        chunks = _scroll_all(client, "EditorialChunk")
    finally:
        client.close()

    games_by_uuid: Dict[str, Dict[str, Any]] = {g["_id"]: g for g in games}

    # --------------------------------------------------
    # Per-game chunk counts
    # --------------------------------------------------
    chunk_counts: Dict[str, int] = defaultdict(int)
    orphan_chunks: List[Tuple[str, str]] = []  # (chunk_id, reason)

    for chunk in chunks:
        chunk_id = str(chunk.get("_id"))
        game_uuid = chunk.get("game_uuid")
        parent_uuid = chunk.get("parent_editorial_uuid")

        if not game_uuid:
            orphan_chunks.append((chunk_id, "missing game_uuid"))
            continue
        if not parent_uuid:
            orphan_chunks.append((chunk_id, "missing parent_editorial_uuid"))
            continue
        if game_uuid not in games_by_uuid:
            orphan_chunks.append(
                (chunk_id, f"game_uuid {game_uuid!r} has no matching Game point")
            )
            continue

        chunk_counts[game_uuid] += 1

    # --------------------------------------------------
    # Zero-chunk games (informational, not a violation)
    # --------------------------------------------------
    zero_chunk_games = [
        (g["_id"], g.get("title", "<unknown>"))
        for g in games
        if chunk_counts.get(g["_id"], 0) == 0
    ]

    # --------------------------------------------------
    # Duplicate unified_game_id (identity scheme collision)
    # --------------------------------------------------
    by_unified_id: Dict[str, List[str]] = defaultdict(list)
    for g in games:
        uid = g.get("unified_game_id")
        if uid:
            by_unified_id[uid].append(g["_id"])

    duplicate_unified_ids = {
        uid: uuids for uid, uuids in by_unified_id.items() if len(uuids) > 1
    }

    # --------------------------------------------------
    # Report
    # --------------------------------------------------
    total_chunks = len(chunks)
    total_games = len(games)

    print("=" * 60)
    print("  CORPUS VERIFICATION REPORT")
    print("=" * 60)
    print(f"  Total Game anchors:        {total_games}")
    print(f"  Total EditorialChunk pts:  {total_chunks}")
    print(f"  Games with >0 chunks:      {total_games - len(zero_chunk_games)}")
    print()

    if zero_chunk_games:
        print(f"  Zero-chunk games ({len(zero_chunk_games)}) — informational:")
        for uuid, title in zero_chunk_games:
            print(f"    - {title} ({uuid})")
        print()

    violations = 0

    if orphan_chunks:
        violations += len(orphan_chunks)
        print(f"  ❌ No-orphan violations ({len(orphan_chunks)}):")
        for chunk_id, reason in orphan_chunks[:50]:
            print(f"    - {chunk_id}: {reason}")
        if len(orphan_chunks) > 50:
            print(f"    ... and {len(orphan_chunks) - 50} more")
        print()
    else:
        print("  ✅ No orphan chunks found.")
        print()

    if duplicate_unified_ids:
        violations += len(duplicate_unified_ids)
        print(f"  ❌ Duplicate unified_game_id ({len(duplicate_unified_ids)}):")
        for uid, uuids in duplicate_unified_ids.items():
            print(f"    - {uid!r}: {uuids}")
        print()
    else:
        print("  ✅ No duplicate unified_game_id values.")
        print()

    print("=" * 60)
    if violations:
        print(f"  RESULT: FAIL — {violations} integrity violation(s)")
    else:
        print("  RESULT: PASS")
    print("=" * 60)

    return 1 if violations else 0


def main() -> None:
    sys.exit(verify_corpus())


if __name__ == "__main__":
    main()
