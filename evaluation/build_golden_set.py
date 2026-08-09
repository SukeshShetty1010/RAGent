#!/usr/bin/env python3
"""
evaluation/build_golden_set.py — Golden Set Drafting Script

Drafts evaluation/data/golden_set.jsonl: 50 records sourced directly
from the live Qdrant corpus (20 factual, 10 comparison, 10
listicle/open, 10 unanswerable). `expected_entities` /
`expected_source_titles` come from real indexed payloads, never from
memory. `ground_truth_answer` for the 40 answerable records is drafted
by Groq from the actual retrieved chunk text and written with
`reviewed: false` — a human review pass (see evaluation/data/README
or the golden set itself) flips it to `true` before scoring. Anything
still `reviewed: false` at scoring time is reported as such, not
silently trusted.

Usage:
    python -m evaluation.build_golden_set
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient

from llm.ragent_client import chat_completion_remote

SCROLL_BATCH = 256
SEED = 42
OUT_PATH = Path(__file__).resolve().parent / "data" / "golden_set.jsonl"

# Real, released games verified (below, via a live Qdrant scroll — not
# assumed) to be absent from this corpus's 100 titles.
UNANSWERABLE_ABSENT_CANDIDATES = [
    "Half-Life 2",
    "Portal 2",
    "Stardew Valley",
    "Hollow Knight",
    "Disco Elysium",
    "Celeste",
    "Untitled Goose Game",
    "Outer Wilds",
    "Return of the Obra Dinn",
    "Slay the Spire",
]

# Titles that do not exist as shipped products (the "GTA VI trap").
UNANSWERABLE_UNRELEASED = [
    "Grand Theft Auto VI",
    "The Elder Scrolls VI",
    "Beyond Good and Evil 2",
]

# Not gaming queries at all — corpus can't possibly answer these.
UNANSWERABLE_OUT_OF_DOMAIN = [
    "What is the capital of France?",
    "What's a good recipe for chocolate chip cookies?",
    "Who won the 2024 US presidential election?",
    "What is the boiling point of water at sea level?",
]


def _get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    return QdrantClient(url=url, api_key=api_key or None)


def _scroll_all(client: QdrantClient, collection_name: str) -> List[Any]:
    points: List[Any] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _draft_answer(query: str, chunks: List[Dict[str, str]]) -> str:
    """Draft a reference answer from real chunk content. Fail-soft:
    an empty string here just means the record needs closer human
    review, not a crashed build."""
    context = "\n\n".join(
        f"[Source: '{c['source_title']}']\n{c['content']}" for c in chunks
    )
    prompt = (
        "Answer the question using ONLY the context below. Be factual "
        "and concise (2-4 sentences). This answer is used as a "
        "ground-truth reference for evaluating a RAG system, so do not "
        "speculate beyond the context.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    for attempt in range(2):
        try:
            return chat_completion_remote(
                prompt, max_tokens=300, temperature=0.1
            ).strip()
        except Exception as exc:
            if attempt == 0:
                time.sleep(3)
                continue
            print(f"  ⚠️ Answer draft failed for {query!r}: {exc}")
            return ""
    return ""


def _entities_and_titles(
    game_points: List[Any], chunks_by_game: Dict[str, List[Dict[str, str]]]
):
    entities = [g.payload.get("title", "") for g in game_points]
    titles = set()
    for g in game_points:
        for c in chunks_by_game[g.id]:
            titles.add(c["source_title"])
    return entities, sorted(titles)


def build_golden_set() -> List[Dict[str, Any]]:
    client = _get_qdrant_client()
    try:
        games = _scroll_all(client, "Game")
        chunks = _scroll_all(client, "EditorialChunk")
    finally:
        client.close()

    chunks_by_game: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for ch in chunks:
        gu = ch.payload.get("game_uuid")
        if gu:
            chunks_by_game[gu].append(
                {
                    "content": ch.payload.get("content") or "",
                    "source_title": ch.payload.get("source_title") or "",
                }
            )

    games_with_chunks = [g for g in games if chunks_by_game.get(g.id)]
    corpus_titles_lower = {
        (g.payload.get("title") or "").lower() for g in games
    }

    rng = random.Random(SEED)
    pool = sorted(games_with_chunks, key=lambda g: g.payload.get("title", ""))
    rng.shuffle(pool)

    if len(pool) < 40:
        raise RuntimeError(
            f"Only {len(pool)} games with chunks in corpus — need >=40 "
            "for factual+comparison+listicle sampling."
        )

    records: List[Dict[str, Any]] = []
    idx = 0

    # ---------------- Factual (20) ----------------
    print("Drafting factual records...")
    for g in pool[:20]:
        title = g.payload.get("title", "")
        query = f"When was {title} released?"
        game_chunks = chunks_by_game[g.id][:5]
        entities, titles = _entities_and_titles([g], chunks_by_game)
        idx += 1
        records.append(
            {
                "id": f"g{idx:03d}",
                "query": query,
                "category": "factual",
                "expected_task": "factual",
                "expected_entities": entities,
                "expected_source_titles": titles,
                "ground_truth_answer": _draft_answer(query, game_chunks),
                "should_refuse": False,
                "reviewed": False,
            }
        )
        time.sleep(1.2)  # stay under Groq free-tier rate limits

    # ---------------- Comparison (10 pairs from next 20 games) --------
    print("Drafting comparison records...")
    comparison_games = pool[20:40]
    for i in range(0, 20, 2):
        g1, g2 = comparison_games[i], comparison_games[i + 1]
        t1 = g1.payload.get("title", "")
        t2 = g2.payload.get("title", "")
        query = f"What are the differences between {t1} and {t2}?"
        combined_chunks = chunks_by_game[g1.id][:3] + chunks_by_game[g2.id][:3]
        entities, titles = _entities_and_titles([g1, g2], chunks_by_game)
        idx += 1
        records.append(
            {
                "id": f"g{idx:03d}",
                "query": query,
                "category": "comparison",
                "expected_task": "comparison",
                "expected_entities": entities,
                "expected_source_titles": titles,
                "ground_truth_answer": _draft_answer(query, combined_chunks),
                "should_refuse": False,
                "reviewed": False,
            }
        )
        time.sleep(1.2)

    # ---------------- Listicle / open (10, from remaining pool) -------
    print("Drafting listicle/open records...")
    listicle_pool = pool[40:] if len(pool[40:]) >= 10 else pool[:10]
    for g in listicle_pool[:10]:
        title = g.payload.get("title", "")
        query = f"What is {title} best known for?"
        game_chunks = chunks_by_game[g.id][:5]
        entities, titles = _entities_and_titles([g], chunks_by_game)
        idx += 1
        records.append(
            {
                "id": f"g{idx:03d}",
                "query": query,
                "category": "listicle",
                "expected_task": "listicle",
                "expected_entities": entities,
                "expected_source_titles": titles,
                "ground_truth_answer": _draft_answer(query, game_chunks),
                "should_refuse": False,
                "reviewed": False,
            }
        )
        time.sleep(1.2)

    # ---------------- Unanswerable (10) --------------------------------
    print("Assembling unanswerable records...")
    absent = [
        t for t in UNANSWERABLE_ABSENT_CANDIDATES
        if t.lower() not in corpus_titles_lower
    ][:4]
    unreleased = [
        t for t in UNANSWERABLE_UNRELEASED
        if t.lower() not in corpus_titles_lower
    ][:3]
    out_of_domain = UNANSWERABLE_OUT_OF_DOMAIN[:3]

    if len(absent) < 4 or len(unreleased) < 3:
        raise RuntimeError(
            "Not enough verified-absent unanswerable candidates — "
            f"absent={len(absent)}, unreleased={len(unreleased)}. "
            "Corpus composition changed; update the candidate lists."
        )

    for title in absent:
        query = f"What's the plot of {title}?"
        idx += 1
        records.append(
            {
                "id": f"g{idx:03d}",
                "query": query,
                "category": "unanswerable",
                "unanswerable_type": "absent_from_corpus",
                "expected_task": "factual",
                "expected_entities": [title],
                "expected_source_titles": [],
                "ground_truth_answer": None,
                "should_refuse": True,
                "reviewed": False,
            }
        )

    for title in unreleased:
        query = f"What platforms can I play {title} on?"
        idx += 1
        records.append(
            {
                "id": f"g{idx:03d}",
                "query": query,
                "category": "unanswerable",
                "unanswerable_type": "unreleased",
                "expected_task": "factual",
                "expected_entities": [title],
                "expected_source_titles": [],
                "ground_truth_answer": None,
                "should_refuse": True,
                "reviewed": False,
            }
        )

    for query in out_of_domain:
        idx += 1
        records.append(
            {
                "id": f"g{idx:03d}",
                "query": query,
                "category": "unanswerable",
                "unanswerable_type": "out_of_domain",
                "expected_task": "factual",
                "expected_entities": [],
                "expected_source_titles": [],
                "ground_truth_answer": None,
                "should_refuse": True,
                "reviewed": False,
            }
        )

    return records


def main() -> None:
    records = build_golden_set()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    refuse_count = sum(1 for r in records if r["should_refuse"])
    unreviewed = sum(1 for r in records if not r["reviewed"])

    print(f"\nWrote {len(records)} records -> {OUT_PATH}")
    print(f"  should_refuse=True: {refuse_count}")
    print(f"  reviewed=false (needs human review before scoring): {unreviewed}")


if __name__ == "__main__":
    main()
