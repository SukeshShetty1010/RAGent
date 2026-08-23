#!/usr/bin/env python3
"""
evaluation/calibrate_relevance.py — Relevance Floor Calibration

retriever/quality_gate.py's per-provider floors (_FLOORS) must come
from measurement, not a guess, because each reranker backend emits a
different scale and none of them documents it reliably:
  - local/hfspace (Xenova/ms-marco-MiniLM-L-6-v2 via fastembed) turned
    out to emit raw logits, roughly -8..+11 on this corpus.
  - cloudflare (@cf/baai/bge-reranker-base) emits 0..1 despite docs
    implying raw logits, heavily saturated toward both ends.
  - voyage (rerank-2.5-lite) emits 0..1.
Run this once per provider and record the result; the output file names
the provider it was produced with, since the numbers are meaningless
without it.

Retrieval-only: no LLM call, no web search, no orchestrator web
decision. Runs every golden_set.jsonl query through RAGRetriever
directly (same "hybrid_rerank" default mode production uses) and the
real CorpusEntityIndex, dumps per-query max/mean rerank_score and the
entity-grounding verdict split by should_refuse, and reports the
max-F1 split point on max_relevance alone (a lower bound: the entity
check in quality_gate.py catches cases this single-signal split
cannot).

Usage:
    python -m evaluation.calibrate_relevance
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from retriever.rag_retriever import RAGRetriever, RERANKER_PROVIDER
from retriever.corpus_index import _get_entity_index

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "data" / "golden_set.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_golden_set(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _f1_at_threshold(records: List[Dict[str, Any]], threshold: float) -> Dict[str, float]:
    """Predict refuse iff max_relevance < threshold; score against should_refuse."""
    tp = fp = fn = tn = 0
    for r in records:
        should_refuse = bool(r["should_refuse"])
        predicted_refuse = r["max_relevance"] < threshold
        if should_refuse and predicted_refuse:
            tp += 1
        elif should_refuse and not predicted_refuse:
            fn += 1
        elif not should_refuse and predicted_refuse:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _floor_candidates(answerable: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """WEAK_FLOOR candidates: every midpoint between adjacent answerable
    max_relevance values, plus the WEAK% each midpoint would produce
    among the answerable group.

    quality_gate.py's derivation methodology (see its _FLOORS comment)
    is "a real gap in the answerable distribution, targeting 10-20%
    WEAK and 0 EMPTY" — done by hand off the raw per_query array for
    every provider calibrated so far. This makes that derivation
    reproducible instead of re-derived per provider. REFUSE_FLOOR is
    not candidate-generated the same way — it is pinned strictly below
    the answerable minimum (see answerable_relevance.min above), not
    chosen from a gap.
    """
    values = sorted(r["max_relevance"] for r in answerable)
    if len(values) < 2:
        return []
    n = len(values)
    candidates = []
    for lo, hi in zip(values, values[1:]):
        if hi == lo:
            continue
        midpoint = (lo + hi) / 2
        weak_count = sum(1 for v in values if v < midpoint)
        candidates.append(
            {
                "midpoint": round(midpoint, 6),
                "gap": round(hi - lo, 6),
                "answerable_weak_pct": round(100 * weak_count / n, 2),
            }
        )
    return candidates


def _best_split(records: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    candidates = sorted({r["max_relevance"] for r in records})
    if not candidates:
        return None
    # Try each observed value as a threshold, plus one above the max
    # (so "always answer" is representable) and 0.0 ("always refuse").
    thresholds = candidates + [candidates[-1] + 1e-6]
    scored = [_f1_at_threshold(records, t) for t in thresholds]
    return max(scored, key=lambda s: s["f1"])


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    values = [r["max_relevance"] for r in records]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    # 6 decimals, not 4: bge-reranker-base's "irrelevant" scores land
    # around 3.7e-05, which 4 decimals rounds to a flat 0.0 and destroys
    # the separation this file exists to measure.
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
    }


def calibrate(golden_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    retriever = RAGRetriever()
    entity_index = _get_entity_index()
    per_query: List[Dict[str, Any]] = []

    try:
        for i, record in enumerate(golden_set, start=1):
            query = record["query"]
            print(f"[{i}/{len(golden_set)}] {query}")

            try:
                chunks = retriever.retrieve(query, limit=5)
            except Exception as exc:
                print(f"  ⚠️ retrieve failed: {exc}")
                per_query.append(
                    {
                        "id": record["id"],
                        "query": query,
                        "should_refuse": record.get("should_refuse"),
                        "error": str(exc),
                    }
                )
                continue

            rerank_scores = [c["rerank_score"] for c in chunks if c.get("rerank_score") is not None]
            entity_grounded = entity_index.assess_grounding(query, chunks)

            if not rerank_scores:
                # No score at all — an empty result set, or a fail-soft
                # rerank omission. Recorded as an error rather than
                # substituted with 0.0: on a 0..1 provider that sentinel
                # is indistinguishable from "scored maximally
                # irrelevant" and would drag the floors down, which is
                # exactly the mistake this run exists to avoid.
                per_query.append(
                    {
                        "id": record["id"],
                        "query": query,
                        "should_refuse": bool(record.get("should_refuse")),
                        "error": "no rerank_score on any chunk",
                        "entity_grounded": entity_grounded,
                        "evidence_count": len(chunks),
                    }
                )
                continue

            per_query.append(
                {
                    "id": record["id"],
                    "query": query,
                    "should_refuse": bool(record.get("should_refuse")),
                    "max_relevance": round(max(rerank_scores), 6),
                    "mean_relevance": round(sum(rerank_scores) / len(rerank_scores), 6),
                    "entity_grounded": entity_grounded,
                    "evidence_count": len(chunks),
                }
            )
    finally:
        retriever.close()

    scored = [r for r in per_query if "error" not in r]
    should_refuse_group = [r for r in scored if r["should_refuse"]]
    answerable_group = [r for r in scored if not r["should_refuse"]]

    best_split = _best_split(scored)

    return {
        "generated": date.today().isoformat(),
        # Which backend produced these numbers. Without it the scores are
        # unreadable — -3.0 is a sane refuse floor for ms-marco logits
        # and nonsense for a 0..1 provider.
        "reranker_provider": RERANKER_PROVIDER,
        "total_queries": len(golden_set),
        "scored": len(scored),
        "errored": len(per_query) - len(scored),
        "should_refuse_relevance": _summarize(should_refuse_group),
        "answerable_relevance": _summarize(answerable_group),
        "best_split_relevance_only": best_split,
        "weak_floor_candidates": _floor_candidates(answerable_group),
        "entity_grounding_on_unanswerable": {
            "false": sum(1 for r in should_refuse_group if r["entity_grounded"] is False),
            "none": sum(1 for r in should_refuse_group if r["entity_grounded"] is None),
            "true": sum(1 for r in should_refuse_group if r["entity_grounded"] is True),
            "total": len(should_refuse_group),
        },
        "per_query": per_query,
    }


def main() -> None:
    golden_set = _load_golden_set(GOLDEN_SET_PATH)
    result = calibrate(golden_set)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Provider goes in the filename so per-provider runs on the same day
    # cannot overwrite each other. The 2026-08-12 baseline predates this
    # and is implicitly "local".
    out_path = RESULTS_DIR / f"relevance_calibration_{RERANKER_PROVIDER}_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {out_path}")
    print(f"Reranker provider: {RERANKER_PROVIDER}")
    print("\n=== SEPARATION ===")
    print(f"should_refuse group max_relevance: {result['should_refuse_relevance']}")
    print(f"answerable group max_relevance:    {result['answerable_relevance']}")
    print(f"\nBest single-signal split (relevance only): {result['best_split_relevance_only']}")
    print(f"\nEntity grounding on should_refuse queries: {result['entity_grounding_on_unanswerable']}")


if __name__ == "__main__":
    main()
