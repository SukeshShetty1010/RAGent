# ingest/chunking.py
"""
Chunking utilities for ingest pipeline.

Input: content docs as plain dicts:
    {"text": "...", "metadata": {...}}

Output: list of chunk dicts with metadata enriched for upsert:
    - doc_type = "chunk"
    - parent_unified_id
    - chunk_index
    - chunk_uuid (8 hex)
    - chunk_type preserved/defaulted
    - language (defaults to en)
    - content_length (int words)
"""

from typing import List, Dict, Any
from data.helper import make_hash8

# Word-based splitter (simple, deterministic)
def _word_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(" ".join(words[i:end]))
        if end == n:
            break
        # next start = end - overlap (but ensure progress)
        i = max(end - overlap, end)
    return chunks


def _chunk_uuid(parent_id: str, idx: int, n: int = 8) -> str:
    base = f"{parent_id}__chunk__{idx}"
    return make_hash8(base)[:n]


def chunk_document(doc: Dict[str, Any], chunk_size: int = 800, chunk_overlap: int = 100, min_words: int = 30) -> List[Dict[str, Any]]:
    text = doc.get("text") or ""
    if not text or not text.strip():
        return []

    meta = dict(doc.get("metadata") or {})
    parent_unified = meta.get("unified_game_id") or meta.get("parent_unified_id") or meta.get("slug") or "unknown"
    chunk_type = meta.get("chunk_type") or "description"
    language = meta.get("language") or "en"

    raw_chunks = _word_chunks(text, chunk_size=chunk_size, overlap=chunk_overlap)
    out: List[Dict[str, Any]] = []
    for idx, ctext in enumerate(raw_chunks):
        words = ctext.split()
        if len(words) < min_words and out:
            # merge with previous chunk to avoid tiny fragments
            prev = out[-1]
            prev_text = prev["text"] + " " + ctext
            prev["text"] = prev_text
            prev_meta = prev["metadata"]
            prev_meta["content_length"] = len(prev_text.split())
            out[-1] = prev
            continue

        cm = dict(meta)
        cm["doc_type"] = "chunk"
        cm["parent_unified_id"] = parent_unified
        cm["chunk_index"] = idx
        cm["chunk_uuid"] = _chunk_uuid(parent_unified, idx)
        cm["chunk_type"] = chunk_type
        cm["language"] = language
        cm["content_length"] = len(ctext.split())
        # ensure raw_source_blob not carried into chunk metadata
        if "raw_source_blob" in cm:
            cm.pop("raw_source_blob", None)

        out.append({"text": ctext, "metadata": cm})

    return out


def chunk_documents(docs: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 100, min_words: int = 30) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for d in docs:
        all_chunks.extend(chunk_document(d, chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_words=min_words))
    return all_chunks
