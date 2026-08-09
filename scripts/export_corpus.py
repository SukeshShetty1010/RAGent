#!/usr/bin/env python3
"""
scripts/export_corpus.py — Full corpus snapshot (durability)

The free-tier Qdrant cluster gets reaped on inactivity (that's what
prompted this whole rebuild). This script dumps every collection to
JSONL under backups/<UTC timestamp>/ — point IDs, payloads, and
vectors — so a future cluster loss costs a restore (scripts/import_corpus.py)
instead of a full re-ingest against RAWG/IGDB/GameSpot/Wikipedia/Steam.

Usage:
    python -m scripts.export_corpus
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient

SCROLL_BATCH = 256

COLLECTIONS = [
    "Game",
    "PlatformSpec",
    "IGDB_Game",
    "GameSpot_Game",
    "EditorialSource",
    "EditorialChunk",
]

BACKUPS_DIR = Path("backups")


def _get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    return QdrantClient(url=url, api_key=api_key or None)


def _serialize_vector(vector: Any) -> Any:
    """Convert a qdrant_client vector (plain list, or named dict mixing
    dense lists with sparse index/value objects) into JSON-safe data."""
    if vector is None:
        return None
    if isinstance(vector, dict):
        out: Dict[str, Any] = {}
        for name, v in vector.items():
            if hasattr(v, "indices") and hasattr(v, "values"):
                out[name] = {"indices": list(v.indices), "values": list(v.values)}
            else:
                out[name] = list(v)
        return out
    return list(vector)


def _export_collection(client: QdrantClient, collection_name: str, out_path: Path) -> int:
    count = 0
    offset = None

    with out_path.open("w", encoding="utf-8") as f:
        while True:
            batch, offset = client.scroll(
                collection_name=collection_name,
                limit=SCROLL_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in batch:
                record = {
                    "id": point.id,
                    "payload": point.payload or {},
                    "vector": _serialize_vector(point.vector),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

            if offset is None:
                break

    return count


def export_corpus() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = BACKUPS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _get_qdrant_client()
    try:
        print(f"Exporting corpus snapshot to {out_dir}/")
        for collection_name in COLLECTIONS:
            out_path = out_dir / f"{collection_name}.jsonl"
            count = _export_collection(client, collection_name, out_path)
            print(f"  {collection_name}: {count} points -> {out_path}")
    finally:
        client.close()

    print(f"\nDone. Restore with: python -m scripts.import_corpus --dir {out_dir}")
    return out_dir


def main() -> None:
    try:
        export_corpus()
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
