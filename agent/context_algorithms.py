# FILE: agent/context_algorithms.py
# Pure Context Assembly Algorithms (NO OBSERVABILITY, NO TASK ROUTING)

from __future__ import annotations

import hashlib
import re
from typing import List, Dict, Any, DefaultDict, Set, Callable
from collections import defaultdict

# ============================================================
# Constants
# ============================================================

# 🔽 Aggressive redundancy detection (matches original behavior)
JACCARD_REDUNDANCY_THRESHOLD = 0.50

# Common stopwords removed during normalization
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "of", "to", "and", "or", "for", "in", "on",
    "with", "by", "from", "that", "this", "as",
}

# ============================================================
# Deduplication
# ============================================================

def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Coarse deduplication using normalized content + source title.
    Preserves the longest surviving chunk per duplicate group.
    """

    normalized_map: Dict[str, Dict[str, Any]] = {}

    for c in chunks:
        content = (c.get("content") or "").strip()
        if not content:
            continue

        source_title = (c.get("source_title") or "").strip()
        norm = normalize_text(content)
        key = hash_text(f"{source_title}::{norm}")

        existing = normalized_map.get(key)
        if not existing or len(content) > len(existing.get("content", "")):
            normalized_map[key] = c

    candidates = list(normalized_map.values())
    survivors: List[Dict[str, Any]] = []

    for c in candidates:
        content = c.get("content", "")
        source = c.get("source_type", "local")

        is_substring = False
        for other in candidates:
            if other is c:
                continue
            if (
                other.get("source_type", "local") == source
                and content in other.get("content", "")
                and len(other.get("content", "")) > len(content)
            ):
                is_substring = True
                break

        if not is_substring:
            survivors.append(c)

    return survivors

# ============================================================
# Ordering Strategies
# ============================================================

def is_fully_reranked(chunks: List[Dict[str, Any]]) -> bool:
    """True when EVERY chunk carries a cross-encoder `rerank_score`.

    All-or-nothing by design. The two fields are not comparable:
    `score` is a Qdrant RRF fusion score for local chunks (~0.016-0.033)
    and Tavily's 0..1 relevance for web chunks, while `rerank_score` is
    the reranker's own scale (raw ms-marco logits for local/hfspace,
    0..1 for cloudflare/voyage). Sorting a list that mixes the two ranks
    chunks by which field they happen to carry rather than by relevance,
    so a partially-scored list falls back to `score` for the WHOLE list.
    """
    return bool(chunks) and all(c.get("rerank_score") is not None for c in chunks)


def relevance_key(chunks: List[Dict[str, Any]]) -> Callable[[Dict[str, Any]], float]:
    """Sort key ranking `chunks` by cross-encoder relevance when every
    chunk has one, else by the original retrieval `score`.

    Decided once over the whole list and reused for every sub-sort
    inside an ordering strategy, so groups never disagree about which
    field they rank on.

    Known limitation: if a local rerank succeeds but score_relevance
    fails for merged web evidence, the list is partial and this falls
    back to `score` for everything -- reintroducing the RRF-vs-Tavily
    cross-scale sort. That is pre-existing behavior, not a regression;
    no comparable signal exists on that path.
    """
    if is_fully_reranked(chunks):
        return lambda c: float(c["rerank_score"])
    return lambda c: float(c.get("score", 0.0) or 0.0)


def order_comparison(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Fair ordering across compared entities:
    - One top chunk per entity first
    - Then general context
    - Then remaining evidence by score
    """

    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    general: List[Dict[str, Any]] = []

    for c in chunks:
        ctx = c.get("retrieval_context")
        if not ctx or ctx == "fallback":
            general.append(c)
        else:
            grouped[ctx].append(c)

    key = relevance_key(chunks)

    for items in grouped.values():
        items.sort(key=key, reverse=True)

    general.sort(key=key, reverse=True)

    ordered: List[Dict[str, Any]] = []

    for items in grouped.values():
        if items:
            ordered.append(items.pop(0))

    remaining: List[Dict[str, Any]] = []
    for items in grouped.values():
        remaining.extend(items)

    remaining.sort(key=key, reverse=True)

    ordered.extend(general)
    ordered.extend(remaining)

    return ordered


def order_listicle(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Preserve explicit list ordering when available,
    otherwise rank by score.
    """

    ordered, unordered = [], []

    for c in chunks:
        idx = c.get("chunk_index", -1)
        if isinstance(idx, int) and idx >= 0:
            ordered.append(c)
        else:
            unordered.append(c)

    ordered.sort(key=lambda c: c.get("chunk_index", 0))
    unordered.sort(key=relevance_key(unordered), reverse=True)

    return ordered + unordered


def order_factual(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Simple relevance ordering for factual queries.
    """
    return sorted(
        chunks,
        key=relevance_key(chunks),
        reverse=True,
    )

# ============================================================
# Character Budget + Redundancy
# ============================================================

def apply_character_budget(
    ordered_chunks: List[Dict[str, Any]],
    *,
    char_cap: int,
) -> tuple[List[Dict[str, Any]], int]:
    """
    Enforce a hard character cap with atomic chunk inclusion
    and Jaccard-based redundancy filtering.

    Returns (final_chunks, redundant_rejections) — the second value
    counts chunks dropped specifically by is_redundant(), separate
    from drops caused by the character cap, so callers can report
    redundancy filtering as its own metric.
    """

    final_chunks: List[Dict[str, Any]] = []
    used_chars = 0
    accepted_token_sets: List[Set[str]] = []
    redundant_rejections = 0

    for c in ordered_chunks:
        content = (c.get("content") or "").strip()
        if not content:
            continue

        content_len = len(content)

        if content_len > char_cap:
            continue

        if used_chars + content_len > char_cap:
            continue

        if is_redundant(content, accepted_token_sets):
            redundant_rejections += 1
            continue

        final_chunks.append(c)
        used_chars += content_len
        accepted_token_sets.append(tokenize(content))

    return final_chunks, redundant_rejections


def is_redundant(
    content: str,
    existing_token_sets: List[Set[str]],
) -> bool:
    """
    Jaccard similarity-based redundancy detection.
    """

    tokens = tokenize(content)
    if not tokens:
        return True

    for existing in existing_token_sets:
        intersection = tokens & existing
        union = tokens | existing
        if not union:
            continue

        if len(intersection) / len(union) >= JACCARD_REDUNDANCY_THRESHOLD:
            return True

    return False

# ============================================================
# Utilities
# ============================================================

def normalize_text(text: str) -> str:
    """
    Aggressive normalization to surface near-duplicates.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = [
        t for t in text.split()
        if t and t not in STOPWORDS
    ]

    return " ".join(tokens)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text: str) -> Set[str]:
    return set(normalize_text(text).split())
