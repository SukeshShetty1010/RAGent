#!/usr/bin/env python3
"""
ingest/chunking.py

FIXED VERSION — eliminates micro-chunking and prioritizes rich text.

Key guarantees:
- Only meaningful long-form text is chunked & embedded
- Metadata is preserved but never embedded alone
- Function signatures are unchanged
- Output schema matches GameChunk exactly
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import pathlib
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# -------------------------
# Constants (CRITICAL)
# -------------------------

MIN_CHUNK_CHARS = 80        # hard floor — no micro chunks
MIN_TEXT_CANDIDATE = 120    # minimum text to even consider chunking

TIER1_KEYS = {
    "description",
    "description_raw",
    "summary",
    "storyline",
    "review",
    "review_text",
    "article_text",
    "body",
    "body_text",
    "content",
    "text",
}

TIER2_KEYS = {
    "title",
    "developer",
    "developers",
    "publisher",
    "publishers",
}

# -------------------------
# Optional tokenizer
# -------------------------

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False


# -------------------------
# Utilities
# -------------------------

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _short_hash(s: str, length: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:length]


def _slugify(s: str, maxlen: int = 32) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:maxlen] or "item"


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)


def _clean(text: str) -> str:
    text = _strip_html(text)
    text = text.replace("\r", "")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# -------------------------
# Token splitting
# -------------------------

def _get_encoder(name: str):
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding(name).encode
    except Exception:
        return None


def split_text(
    text: str,
    encode_fn: Optional[Callable[[str], List[int]]],
    chunk_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    if not encode_fn:
        step = max(1, chunk_tokens * 4 - overlap_tokens * 4)
        return [
            text[i : i + chunk_tokens * 4]
            for i in range(0, len(text), step)
        ]

    tokens = encode_fn(text)
    out = []
    step = max(1, chunk_tokens - overlap_tokens)
    for i in range(0, len(tokens), step):
        slice_ = tokens[i : i + chunk_tokens]
        try:
            out.append(tiktoken.get_encoding("cl100k_base").decode(slice_))
        except Exception:
            pass
    return out


# -------------------------
# Recursive text extraction
# -------------------------

def _collect_tier1_text(obj: Any, path: str = "") -> List[Tuple[str, str]]:
    """
    Recursively extract Tier-1 rich text fields.
    Returns list of (source_path, text)
    """
    found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            lk = k.lower()
            if lk in TIER1_KEYS and isinstance(v, str) and len(v) >= MIN_TEXT_CANDIDATE:
                found.append((new_path, v))
            else:
                found.extend(_collect_tier1_text(v, new_path))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_collect_tier1_text(v, f"{path}[{i}]"))

    return found


# -------------------------
# Chunk builder (CORE)
# -------------------------

def build_chunks_from_merged(
    merged_obj: Dict[str, Any],
    chunk_tokens: int = 800,
    overlap_tokens: int = 200,
    model_encoding: str = "cl100k_base",
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:

    unified_id = merged_obj.get("unified_id") or f"gen:{_short_hash(json.dumps(merged_obj)[:200])}"
    title = merged_obj.get("title", "")
    encode_fn = _get_encoder(model_encoding)

    tier1_texts = _collect_tier1_text(merged_obj)

    if not tier1_texts:
        # absolute fallback — only if NOTHING rich exists
        desc = merged_obj.get("description", "")
        if isinstance(desc, str) and len(desc) >= MIN_TEXT_CANDIDATE:
            tier1_texts = [("description", desc)]

    chunks: List[Dict[str, Any]] = []

    for doc_idx, (src_path, raw_text) in enumerate(tier1_texts):
        text = _clean(raw_text)

        for chunk_idx, chunk_text in enumerate(
            split_text(text, encode_fn, chunk_tokens, overlap_tokens)
        ):
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue

            # Title is CONTEXT, not standalone chunk
            full_text = f"{title}\n\n{chunk_text}".strip()

            chunk_id = f"{_slugify(src_path)}_{_slugify(unified_id)}_{chunk_idx}_{_short_hash(full_text,6)}"

            meta = {
                "unified_id": unified_id,
                "doc_id": f"{unified_id}::{src_path}",
                "source": "merged",
                "title": title,
                "chunk_index": chunk_idx,
                "char_length": len(full_text),
                "release_date": merged_obj.get("release_date"),
                "release_year": merged_obj.get("release_year"),
                "platforms": merged_obj.get("platforms") or [],
                "genres": merged_obj.get("genres") or [],
                "developers": merged_obj.get("developers") or [],
                "publishers": merged_obj.get("publishers") or [],
                "created_at": _now_iso(),
                "namespace": namespace,
            }

            chunks.append({
                "id": chunk_id,
                "text": full_text,
                "meta": meta,
            })

    return chunks


# -------------------------
# Save helpers
# -------------------------

def save_chunks_jsonl(chunks: List[Dict[str, Any]], out_path: str) -> None:
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# -------------------------
# CLI
# -------------------------

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    merged = json.loads(pathlib.Path(args.merged).read_text(encoding="utf-8"))
    chunks = build_chunks_from_merged(merged)

    out = pathlib.Path(args.outdir) / "chunks.jsonl"
    save_chunks_jsonl(chunks, str(out))
    print(f"[saved] {len(chunks)} chunks → {out}")


if __name__ == "__main__":
    _cli()
