#!/usr/bin/env python3
"""
evaluation/measure_noise_drop_rate.py — T26: real-corpus noise drop rate

T18 (retriever/quality_gate.py) narrowed the noise filter from "any
keyword match anywhere in title+content" to a source-field match
(title/url) OR a >=3-distinct-keyword content-density signal, plus
regression tests proving the change fixes incidental-mention false
positives ("a great deal of freedom"). What T18 never measured is the
actual drop-rate delta against the real corpus, per keyword.

This is a one-time measurement against the live EditorialChunk
collection. It imports and calls the REAL, live
RetrievalQualityGate.is_noise() for the new-rule verdict (production
parity, never re-derived) and reimplements the retired pre-T18 rule
locally for comparison — that rule no longer exists in the codebase.

retriever/quality_gate.py is read-only here: never modified.

Usage:
    python -m evaluation.measure_noise_drop_rate
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

from retriever.quality_gate import RetrievalQualityGate

COLLECTION = "EditorialChunk"
SCROLL_BATCH = 200
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MAX_SAMPLES = 15


def _get_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    return QdrantClient(url=url, api_key=api_key or None, timeout=60)


def _old_rule_hits(title: str, content: str, keywords: set) -> set:
    """Pre-T18, retired logic, for measurement only (commit eb500fd).

    The old rule concatenated title+content only (never url) and
    flagged noise on ANY single keyword match, with no density
    requirement.
    """
    text = f"{title} {content}".lower()
    return {k for k in keywords if re.search(rf"\b{re.escape(k)}\b", text)}


def _new_rule_reason(gate: RetrievalQualityGate, title: str, content: str, url: str) -> Dict[str, Any]:
    """Reproduce is_noise()'s own short-circuit order so a drop can be
    attributed to source_match vs content_density. Includes a drift
    guard: if content_density is claimed, the actual distinct-hit count
    must be >= CONTENT_NOISE_DENSITY, or is_noise()'s internals have
    changed without this script being updated.
    """
    keywords = gate.SOURCE_NOISE_KEYWORDS
    source_text = f"{title} {url}".lower()
    source_hits = {k for k in keywords if re.search(rf"\b{re.escape(k)}\b", source_text)}
    if source_hits:
        assert gate.is_noise(title, content, url) is True, "drift: source match but is_noise()=False"
        return {"is_noise": True, "reason": "source_match", "keywords": source_hits}

    content_text = content.lower()
    content_hits = {k for k in keywords if re.search(rf"\b{re.escape(k)}\b", content_text)}
    is_dense = len(content_hits) >= gate.CONTENT_NOISE_DENSITY
    actual = gate.is_noise(title, content, url)
    if is_dense:
        assert len(content_hits) >= gate.CONTENT_NOISE_DENSITY, "drift: density claimed below threshold"
        assert actual is True, "drift: density match but is_noise()=False"
        return {"is_noise": True, "reason": "content_density", "keywords": content_hits}

    assert actual is False, "drift: neither signal fired but is_noise()=True"
    return {"is_noise": False, "reason": None, "keywords": set()}


def measure() -> Dict[str, Any]:
    client = _get_client()
    gate = RetrievalQualityGate()
    keywords = sorted(gate.SOURCE_NOISE_KEYWORDS)

    total_scanned = 0
    old_drop_count = 0
    new_rule_by_reason = {"source_match": 0, "content_density": 0}
    both_drop = recovered = new_only = both_keep = 0

    per_keyword = {
        k: {"keyword": k, "old_rule_drop_count": 0, "new_rule_source_match_count": 0, "new_rule_content_density_count": 0}
        for k in keywords
    }

    samples_recovered: List[Dict[str, Any]] = []
    samples_new_only: List[Dict[str, Any]] = []

    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=["content", "source_title", "source_url"],
            with_vectors=False,
        )
        if not records:
            break

        for rec in records:
            payload = rec.payload or {}
            title = (payload.get("source_title") or "")
            content = (payload.get("content") or "")
            url = (payload.get("source_url") or "")

            total_scanned += 1

            old_hits = _old_rule_hits(title, content, gate.SOURCE_NOISE_KEYWORDS)
            is_old = bool(old_hits)
            for k in old_hits:
                per_keyword[k]["old_rule_drop_count"] += 1

            new_verdict = _new_rule_reason(gate, title, content, url)
            is_new = new_verdict["is_noise"]
            if new_verdict["reason"] == "source_match":
                for k in new_verdict["keywords"]:
                    per_keyword[k]["new_rule_source_match_count"] += 1
            elif new_verdict["reason"] == "content_density":
                for k in new_verdict["keywords"]:
                    per_keyword[k]["new_rule_content_density_count"] += 1

            if is_old:
                old_drop_count += 1
            if is_new:
                new_rule_by_reason[new_verdict["reason"]] += 1

            sample = {
                "id": str(rec.id),
                "title": title,
                "url": url,
                "content_snippet": content[:200],
            }
            if is_old and is_new:
                both_drop += 1
            elif is_old and not is_new:
                recovered += 1
                if len(samples_recovered) < MAX_SAMPLES:
                    samples_recovered.append(sample)
            elif is_new and not is_old:
                new_only += 1
                if len(samples_new_only) < MAX_SAMPLES:
                    samples_new_only.append(sample)
            else:
                both_keep += 1

        if offset is None:
            break

    new_drop_count = both_drop + new_only
    qdrant_total = client.count(COLLECTION, exact=True).count

    sanity = {
        "scanned_matches_qdrant_count": total_scanned == qdrant_total,
        "both_drop_plus_recovered_equals_old_drop": (both_drop + recovered) == old_drop_count,
        "both_drop_plus_new_only_equals_new_drop": (both_drop + new_only) == new_drop_count,
    }
    assert sanity["scanned_matches_qdrant_count"], f"scanned {total_scanned} != qdrant count {qdrant_total}"
    assert sanity["both_drop_plus_recovered_equals_old_drop"]
    assert sanity["both_drop_plus_new_only_equals_new_drop"]

    def pct(n: int) -> float:
        return round(100 * n / total_scanned, 4) if total_scanned else 0.0

    return {
        "generated": date.today().isoformat(),
        "collection": COLLECTION,
        "total_chunks_scanned": total_scanned,
        "qdrant_total_count": qdrant_total,
        "keyword_set": keywords,
        "content_noise_density_threshold": gate.CONTENT_NOISE_DENSITY,
        "summary": {
            "old_rule_drop_count": old_drop_count,
            "old_rule_drop_pct": pct(old_drop_count),
            "new_rule_drop_count": new_drop_count,
            "new_rule_drop_pct": pct(new_drop_count),
            "new_rule_by_reason": new_rule_by_reason,
            "recovered_by_t18": {
                "count": recovered,
                "pct_of_total": pct(recovered),
                "description": "old=True, new=False -- chunks T18 saved",
            },
            "new_only_drops": {
                "count": new_only,
                "pct_of_total": pct(new_only),
                "description": "new=True, old=False -- url-only match, old rule never saw url",
            },
            "both_drop_count": both_drop,
            "both_keep_count": both_keep,
        },
        "per_keyword": [per_keyword[k] for k in keywords],
        "samples": {
            "recovered_by_t18": samples_recovered,
            "new_only_drops": samples_new_only,
        },
        "sanity_checks": sanity,
    }


def main() -> None:
    result = measure()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"noise_drop_rate_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {out_path}")
    print(f"Scanned {result['total_chunks_scanned']} chunks (qdrant count: {result['qdrant_total_count']})")
    print(f"Old rule drops: {result['summary']['old_rule_drop_count']} ({result['summary']['old_rule_drop_pct']}%)")
    print(f"New rule drops: {result['summary']['new_rule_drop_count']} ({result['summary']['new_rule_drop_pct']}%)")
    print(f"Recovered by T18: {result['summary']['recovered_by_t18']}")
    print(f"New-only drops (url-only trigger): {result['summary']['new_only_drops']}")
    print(f"Sanity checks: {result['sanity_checks']}")


if __name__ == "__main__":
    main()
