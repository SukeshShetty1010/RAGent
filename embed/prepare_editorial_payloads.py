from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from typing import Dict, List, Tuple
from uuid import UUID, uuid5

from chunking.editorial_chunker import EditorialChunker
from ingest.gamespot_editorial_normalize import fetch_and_prepare_gamespot

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
EDITORIAL_CHUNK_NAMESPACE = UUID("87654321-4321-6789-4321-678987654321")
MAX_WORDS_PER_CHUNK = 1000
HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_\-]", "", name.strip().lower().replace(" ", "_"))


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Quality Filter
# ---------------------------------------------------------------------
class QualityFilter:
    def __init__(self, max_price_density: float = 0.01, min_avg_line_length: int = 30):
        self.max_price_density = max_price_density
        self.min_avg_line_length = min_avg_line_length

    def evaluate(self, content: str) -> Tuple[bool, str]:
        if self._high_price_density(content):
            return False, "high price density"
        if self._list_like_structure(content):
            return False, "list-like structure"
        return True, ""

    def _high_price_density(self, content: str) -> bool:
        symbols = sum(content.count(s) for s in ("$", "€", "£"))
        return symbols / max(len(content), 1) > self.max_price_density

    def _list_like_structure(self, content: str) -> bool:
        lines = [l for l in content.splitlines() if l.strip()]
        if not lines:
            return True
        avg_len = sum(len(l) for l in lines) / len(lines)
        return avg_len < self.min_avg_line_length


# ---------------------------------------------------------------------
# Core Generator
# ---------------------------------------------------------------------
def generate_chunk_payloads(game_name: str, canonical_uuid: str) -> List[Dict]:
    gamespot_payload = fetch_and_prepare_gamespot(
        game_name=game_name,
        canonical_game_uuid=canonical_uuid,
    )

    if not gamespot_payload:
        raise RuntimeError("No GameSpot data available")

    gamespot_uuid = gamespot_payload["uuid"]
    editorial_object = gamespot_payload["properties"]

    chunker = EditorialChunker(chunk_size=300, overlap=50)
    raw_chunks = chunker.process_game_editorial(
        editorial_object=editorial_object,
        game_uuid=canonical_uuid,
        gamespot_uuid=gamespot_uuid,
    )

    quality_filter = QualityFilter()

    total = len(raw_chunks)
    dropped_quality = 0
    dropped_safety = 0
    dropped_html = 0
    objects: List[Dict] = []

    for chunk in raw_chunks:
        chunk_uuid = uuid5(EDITORIAL_CHUNK_NAMESPACE, chunk["chunk_id"])
        content = chunk["content"]

        # ---------------- HTML SAFETY CHECK ----------------
        if HTML_TAG_RE.search(content):
            dropped_html += 1
            logger.critical(
                "[Data Quality] Chunk %s contains HTML tags. Dropping.",
                chunk_uuid,
            )
            continue

        # ---------------- SIZE SAFETY ----------------
        words = len(content.split())
        if words > MAX_WORDS_PER_CHUNK:
            dropped_safety += 1
            logger.warning(
                "[Safety] Dropping oversized chunk %s (Words: %d)",
                chunk_uuid,
                words,
            )
            continue

        ok, reason = quality_filter.evaluate(content)
        if not ok:
            dropped_quality += 1
            logger.warning("Dropped chunk %s due to %s", chunk_uuid, reason)
            continue

        objects.append(
            {
                "uuid": str(chunk_uuid),
                "class": "EditorialChunk",
                "properties": {
                    "content": content,
                    "content_hash": _hash_content(content),
                    "source_title": chunk["source_title"],
                    "source": chunk["source"],
                    "content_type": chunk["content_type"],
                    "chunk_index": chunk["chunk_index"],
                    "game": {
                        "beacon": f"weaviate://localhost/Game/{canonical_uuid}"
                    },
                    "parent_editorial": {
                        "beacon": f"weaviate://localhost/GameSpot_Game/{gamespot_uuid}"
                    },
                },
            }
        )

    logger.info("Total Chunks Generated: %d", total)
    logger.info("Dropped (HTML): %d", dropped_html)
    logger.info("Dropped (Quality): %d", dropped_quality)
    logger.info("Dropped (Safety): %d", dropped_safety)
    logger.info("Final Payloads: %d", len(objects))

    return objects


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--uuid", required=True)
    args = parser.parse_args()

    payloads = generate_chunk_payloads(args.game, args.uuid)

    os.makedirs("data", exist_ok=True)
    fname = f"data/{_safe_name(args.game)}_editorial_chunk_payloads.json"

    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d payloads to %s", len(payloads), fname)
