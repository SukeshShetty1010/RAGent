#!/usr/bin/env python3
"""
ingest/chunking.py

Chunking utility for the RAG pipeline.

This file contains the chunking logic (token/char splitters, text cleaning,
chunk builders, dedupe, and save helpers) and a CLI that supports two modes:

1) --merged <path>
   Read an existing merged JSON (output of merge.py) and produce chunks from it.

2) (no --merged) interactive / --game <name>
   Calls the loader (ingest.loader.fetch_all_sources) and merge (ingest.merge)
   helpers to fetch and merge live data, then chunk that merged object.

This keeps the chunking functions pure (no I/O) while the CLI demonstrates
end-to-end behaviour when run standalone.

Note: this file intentionally keeps the chunking code self-contained so it
does not depend on external tokenizers at import time (tiktoken is optional).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import os
import pathlib
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Optional token encoder (tiktoken) is used when available.
try:
    import tiktoken  # type: ignore
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

# Import fetch/merge helpers from the project's ingest modules.
# These imports are local to the CLI flow so importing this module in other
# contexts will not attempt to call network operations.
try:
    from ingest.loader import fetch_all_sources, _safe_name  # type: ignore
    from ingest.merge import _normalize_loader_payloads, merge_three_sources  # type: ignore
    _HAS_REMOTE_HELPERS = True
except Exception:
    fetch_all_sources = None  # type: ignore
    _safe_name = lambda s: re.sub(r"[^a-z0-9_\-]", "", s.strip().lower().replace(" ", "_"))
    _normalize_loader_payloads = None  # type: ignore
    merge_three_sources = None  # type: ignore
    _HAS_REMOTE_HELPERS = False


# -------------------------
# Utilities
# -------------------------

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _short_hash(s: str, length: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:length]


def _slugify(s: str, maxlen: int = 32) -> str:
    if not s:
        return "no_title"
    s2 = s.strip().lower()
    s2 = re.sub(r"[^\w\s-]", "", s2)
    s2 = re.sub(r"[\s_-]+", "_", s2)
    return s2[:maxlen]


def _clean_newlines(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\\n", "\n").replace("\\r", "\n")
    text = html.unescape(text)
    text = text.replace("\r", "")
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


def _strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text


# -------------------------
# Deduplication
# -------------------------

def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict

    def normalize_text(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\\n", "\n")
        s = re.sub(r"\s+", " ", s)
        return s.strip().lower()

    hash_to_indices = defaultdict(list)
    for idx, ch in enumerate(chunks):
        text = ch.get("text") or ""
        norm = normalize_text(text)
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        hash_to_indices[h].append(idx)

    deduped = []
    for h, idx_list in hash_to_indices.items():
        first_idx = idx_list[0]
        first_chunk = chunks[first_idx]
        meta = dict(first_chunk.get("meta") or {})
        meta["content_hash"] = h
        if len(idx_list) > 1:
            dup_info = []
            for i in idx_list[1:]:
                dup = chunks[i]
                dup_info.append({
                    "id": dup.get("id"),
                    "doc_id": dup.get("meta", {}).get("doc_id"),
                    "source": dup.get("meta", {}).get("source"),
                })
            meta["duplicate_count"] = len(idx_list) - 1
            meta["duplicate_sources"] = dup_info
        first_chunk["meta"] = meta
        deduped.append(first_chunk)
    return deduped


# -------------------------
# Text extraction from merged object
# -------------------------

def _gather_text_items(merged: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    unified_id = merged.get("unified_id") or f"gen:{_short_hash(json.dumps(merged)[:200])}"
    title = merged.get("title") or ""
    desc = merged.get("description") or ""

    if title:
        items.append({
            "doc_idx": 0,
            "doc_id": f"{unified_id}::title",
            "source": "merged:title",
            "title": title,
            "text": _clean_newlines(title),
            "meta": {"unified_id": unified_id, "section": "title"},
        })

    if desc:
        items.append({
            "doc_idx": 1,
            "doc_id": f"{unified_id}::description",
            "source": "merged:description",
            "title": (f"{title} — description" if title else "description"),
            "text": _clean_newlines(desc),
            "meta": {"unified_id": unified_id, "section": "description"},
        })

    docs = merged.get("documents") or []
    base_doc_idx = len(items)
    for i, raw in enumerate(docs):
        doc_idx = base_doc_idx + i
        text_candidates = []
        doc_id = raw.get("id") or f"{unified_id}::doc::{i}"
        source = raw.get("source") or "merged:document"
        title_cand = raw.get("title")
        for fn in ("content", "body", "text", "excerpt", "summary", "deck", "review_text", "article_text"):
            v = raw.get(fn)
            if isinstance(v, str) and v.strip():
                text_candidates.append((fn, v))
        if not text_candidates:
            for k, v in raw.items():
                if isinstance(v, str) and len(v) > 60:
                    text_candidates.append((k, v))
        if not text_candidates:
            continue
        text_candidates.sort(key=lambda t: len(t[1]), reverse=True)
        chosen_field, chosen_text = text_candidates[0]
        chosen_text = _strip_html(chosen_text)
        chosen_text = _clean_newlines(chosen_text)
        items.append({
            "doc_idx": doc_idx,
            "doc_id": doc_id,
            "source": source,
            "title": title_cand or f"{title} — doc {i}",
            "text": chosen_text,
            "meta": {"unified_id": unified_id, "document_field": chosen_field, "document_meta": raw.get("meta") or {}},
        })

    meta_lists = {
        "platforms": merged.get("platforms"),
        "genres": merged.get("genres"),
        "developers": merged.get("developers"),
        "publishers": merged.get("publishers"),
    }
    mbase = len(items)
    for k, lst in meta_lists.items():
        if lst:
            txt = ", ".join(str(x) for x in lst)
            items.append({
                "doc_idx": mbase,
                "doc_id": f"{unified_id}::meta::{k}",
                "source": f"merged:meta:{k}",
                "title": (f"{title} — {k}" if title else k),
                "text": _clean_newlines(txt),
                "meta": {"unified_id": unified_id, "section": f"meta:{k}"},
            })
            mbase += 1

    return items


# -------------------------
# Tokenizer helpers & splitters
# -------------------------

def _get_token_encoder(encoding_name: str = "cl100k_base"):
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return enc.encode
    except Exception:
        try:
            enc = tiktoken.encoding_for_model(encoding_name)
            return enc.encode
        except Exception:
            return None


def split_by_chars(text: str, chunk_size_chars: int = 1000, overlap: int = 200) -> List[Tuple[int, int, str]]:
    if not text:
        return []
    text = text.strip()
    length = len(text)
    step = max(1, chunk_size_chars - overlap)
    chunks = []
    idx = 0
    start = 0
    while start < length:
        end = min(length, start + chunk_size_chars)
        chunk = text[start:end]
        chunks.append((idx, start, chunk))
        idx += 1
        start += step
    return chunks


def split_by_tokens(text: str, encode_fn: Optional[Callable[[str], List[int]]], chunk_tokens: int = 800, overlap_tokens: int = 200) -> List[Tuple[int, int, str]]:
    if not text:
        return []
    if encode_fn is None:
        approx_chars = max(200, int(chunk_tokens * 4))
        approx_overlap = int(overlap_tokens * 4)
        return split_by_chars(text, approx_chars, approx_overlap)

    token_ids = encode_fn(text)
    total_tokens = len(token_ids)
    step = max(1, chunk_tokens - overlap_tokens)
    chunks = []
    start = 0
    idx = 0
    try:
        enc = tiktoken.get_encoding("cl100k_base") if TIKTOKEN_AVAILABLE else None
        while start < total_tokens:
            end = min(total_tokens, start + chunk_tokens)
            token_slice = token_ids[start:end]
            try:
                chunk_text = enc.decode(token_slice) if enc is not None else ""
            except Exception:
                char_start = int(len(text) * (start / max(1, total_tokens)))
                char_end = int(len(text) * (end / max(1, total_tokens)))
                chunk_text = text[char_start:char_end]
            chunks.append((idx, start, chunk_text))
            idx += 1
            start += step
    except Exception:
        return split_by_chars(text, chunk_size_chars=chunk_tokens * 4, overlap=overlap_tokens * 4)
    return chunks


# -------------------------
# Build chunk objects
# -------------------------

def _build_short_chunk_id(source: str, unified_id: str, doc_idx: int, chunk_idx: int, chunk_text: str) -> str:
    s = re.sub(r"[^\w\-]", "", str(source or "src")).lower()[:16]
    u = _slugify(str(unified_id), maxlen=16)
    h = _short_hash(chunk_text, length=6)
    return f"{s}_{u}_d{doc_idx}_c{chunk_idx}_{h}"


def build_chunks_from_merged(
    merged_obj: Dict[str, Any],
    chunk_tokens: int = 800,
    overlap_tokens: int = 200,
    model_encoding: str = "cl100k_base",
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    items = _gather_text_items(merged_obj)
    unified_id = merged_obj.get("unified_id") or f"gen:{_short_hash(json.dumps(merged_obj)[:200])}"
    encode_fn = _get_token_encoder(model_encoding)

    out_chunks: List[Dict[str, Any]] = []
    for it in items:
        doc_idx = int(it.get("doc_idx", 0))
        doc_id = it.get("doc_id") or f"{unified_id}::doc::{doc_idx}"
        src = it.get("source") or "merged"
        title = it.get("title") or ""
        text = str(it.get("text") or "")
        if not text.strip():
            continue
        text = _strip_html(text)
        text = _clean_newlines(text)
        token_chunks = split_by_tokens(text, encode_fn, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        for chunk_idx, start_token_or_char_idx, chunk_text in token_chunks:
            short_id = _build_short_chunk_id(src, unified_id, doc_idx, chunk_idx, chunk_text)
            meta = {
                "unified_id": unified_id,
                "doc_id": doc_id,
                "source": src,
                "title": title,
                "chunk_index": chunk_idx,
                "start_token_or_char": start_token_or_char_idx,
                "char_length": len(chunk_text),
                "release_date": merged_obj.get("release_date"),
                "release_year": merged_obj.get("release_year"),
                "platforms": merged_obj.get("platforms") or [],
                "genres": merged_obj.get("genres") or [],
                "created_at": _now_iso(),
                "namespace": namespace,
            }
            out_chunks.append({"id": short_id, "text": chunk_text, "meta": meta})
    out_chunks = _dedupe_chunks(out_chunks)
    return out_chunks


# -------------------------
# Save helpers
# -------------------------

def save_chunks_jsonl(chunks: List[Dict[str, Any]], out_path: str) -> None:
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")


def save_chunks_json(chunks: List[Dict[str, Any]], out_path: str) -> None:
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# CLI / Main
# -------------------------

def _cli():
    p = argparse.ArgumentParser(prog="ingest.chunking", description="Create RAG chunks from merged game JSON or live fetch")
    p.add_argument("--merged", "-m", help="Path to merged JSON file (output of merge.py). If omitted, --game will be used to fetch+merge live data.")
    p.add_argument("--game", "-g", help="Game name to fetch (used when --merged is omitted)")
    p.add_argument("--outdir", "-o", default=".", help="Output directory")
    p.add_argument("--chunk-tokens", type=int, default=800, help="Chunk size in tokens (used when tiktoken available)")
    p.add_argument("--overlap-tokens", type=int, default=200, help="Overlap size in tokens")
    p.add_argument("--model-encoding", default="cl100k_base", help="tiktoken encoding name (default cl100k_base)")
    p.add_argument("--jsonl", action="store_true", help="Write chunks as JSONL (one JSON object per line)")
    p.add_argument("--json", action="store_true", help="Write chunks as single JSON array (chunks.json)")
    p.add_argument("--namespace", default=None, help="Optional namespace for vector DB")
    args = p.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged_obj = None

    if args.merged:
        merged_path = pathlib.Path(args.merged)
        if not merged_path.exists():
            print(f"ERROR: merged file not found: {merged_path}", file=sys.stderr)
            sys.exit(2)
        with merged_path.open("r", encoding="utf-8") as fh:
            merged_obj = json.load(fh)
    else:
        game = args.game
        if not game:
            try:
                game = input("Enter game name to fetch & merge: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                return
        if not game:
            print("Game name required.")
            return

        if not _HAS_REMOTE_HELPERS:
            print("ERROR: ingest.loader or ingest.merge helpers are not importable.\n"
                  "Make sure you run this from the project root so Python can import ingest.* modules.")
            sys.exit(3)

        print(f"[fetch] Pulling RAWG / IGDB / GameSpot data for: {game!r}")
        results = fetch_all_sources(game, strip_visual=True)  # type: ignore
        wrapped = _normalize_loader_payloads(results)  # type: ignore
        merged_obj = merge_three_sources(wrapped.get("rawg"), wrapped.get("igdb"), wrapped.get("gamespot"))  # type: ignore

        # save merged to outdir for reproducibility
        safe = _safe_name(game)  # type: ignore
        merged_path = outdir / f"{safe}_merged.json"
        with merged_path.open("w", encoding="utf-8") as fh:
            json.dump(merged_obj, fh, indent=2, ensure_ascii=False, default=str)
        print(f"[saved] merged JSON -> {merged_path}")

    if TIKTOKEN_AVAILABLE:
        print("[info] tiktoken available: token-based splitting enabled")
    else:
        print("[info] tiktoken NOT available: using fallback char-based splitting (approx tokens -> chars)")

    chunks = build_chunks_from_merged(
        merged_obj,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        model_encoding=args.model_encoding,
        namespace=args.namespace,
    )

    base = _slugify((getattr(merged_obj, "title", None) or merged_obj.get("title") or "merged"))
    if args.jsonl:
        out_path = outdir / f"{base}_chunks.jsonl"
        save_chunks_jsonl(chunks, str(out_path))
        print(f"[saved] {len(chunks)} chunks -> {out_path}")
    if args.json:
        out_path = outdir / f"{base}_chunks.json"
        save_chunks_json(chunks, str(out_path))
        print(f"[saved] {len(chunks)} chunks -> {out_path}")
    if not args.json and not args.jsonl:
        out_path = outdir / f"{base}_chunks.jsonl"
        save_chunks_jsonl(chunks, str(out_path))
        print(f"[saved] {len(chunks)} chunks -> {out_path}")

    # Print small sample
    print("\nSample chunk (first):")
    if chunks:
        s = chunks[0]
        print("ID:", s.get("id"))
        print("META:", {k: s.get("meta", {}).get(k) for k in ("unified_id", "doc_id", "chunk_index", "char_length")})
        print("TEXT (first 300 chars):")
        print((s.get("text") or "")[:300].replace("\n", " "))


if __name__ == "__main__":
    _cli()