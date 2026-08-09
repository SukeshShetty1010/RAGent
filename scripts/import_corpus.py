#!/usr/bin/env python3
"""
scripts/import_corpus.py — Restore a corpus snapshot (durability)

Restores a snapshot written by scripts/export_corpus.py. Point IDs,
payloads, and vectors are preserved byte-for-byte, so a restore is a
straight replay — no upstream API (RAWG/IGDB/GameSpot/Wikipedia/Steam)
is touched.

Usage:
    python -m scripts.import_corpus --dir backups/20260809T120000Z
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from vector.create_schema import create_all_collections

UPSERT_BATCH = 64

COLLECTIONS = [
    "Game",
    "PlatformSpec",
    "IGDB_Game",
    "GameSpot_Game",
    "EditorialSource",
    "EditorialChunk",
]


def _get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    # Restoring 768-dim dense + BM25 sparse vectors in bulk against a
    # remote cluster needs more headroom than the client's short default
    # read timeout — that's what tripped on the first EditorialChunk batch.
    return QdrantClient(url=url, api_key=api_key or None, timeout=60)


def _deserialize_vector(vector: Any) -> Any:
    """Inverse of export_corpus._serialize_vector: rebuild SparseVector
    objects for named sparse components (e.g. EditorialChunk's "bm25")."""
    if vector is None:
        return None
    if isinstance(vector, dict):
        out: Dict[str, Any] = {}
        for name, v in vector.items():
            if isinstance(v, dict) and "indices" in v and "values" in v:
                out[name] = SparseVector(indices=v["indices"], values=v["values"])
            else:
                out[name] = v
        return out
    return vector


def _import_collection(client: QdrantClient, collection_name: str, in_path: Path) -> int:
    if not in_path.exists():
        print(f"  {collection_name}: no snapshot file, skipping")
        return 0

    count = 0
    batch: list[PointStruct] = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            batch.append(
                PointStruct(
                    id=record["id"],
                    vector=_deserialize_vector(record["vector"]),
                    payload=record["payload"],
                )
            )
            if len(batch) >= UPSERT_BATCH:
                client.upsert(collection_name=collection_name, points=batch)
                count += len(batch)
                batch = []

    if batch:
        client.upsert(collection_name=collection_name, points=batch)
        count += len(batch)

    return count


def import_corpus(backup_dir: Path) -> None:
    if not backup_dir.exists():
        raise FileNotFoundError(f"Backup directory not found: {backup_dir}")

    client = _get_qdrant_client()
    try:
        print("Ensuring collections exist...")
        create_all_collections(client)

        print(f"\nRestoring corpus snapshot from {backup_dir}/")
        for collection_name in COLLECTIONS:
            in_path = backup_dir / f"{collection_name}.jsonl"
            count = _import_collection(client, collection_name, in_path)
            print(f"  {collection_name}: {count} points restored")
    finally:
        client.close()

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a corpus snapshot into Qdrant")
    parser.add_argument(
        "--dir",
        required=True,
        help="Backup directory (e.g. backups/20260809T120000Z, from export_corpus.py)",
    )
    args = parser.parse_args()

    try:
        import_corpus(Path(args.dir))
    except Exception as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
