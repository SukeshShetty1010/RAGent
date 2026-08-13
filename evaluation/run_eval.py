#!/usr/bin/env python3
"""
evaluation/run_eval.py — Golden Set Run Harness

Drives RageEngine.run() over evaluation/data/golden_set.jsonl — the
exact same execution path production uses — and appends one JSON
record per query to evaluation/results/runs_<date>[_corpusonly].jsonl.

`merge_state` is recorded per-record because retriever/orchestrator.py
can rescue an out-of-corpus query with a web search (QUALITY_EMPTY ->
unconditional web attempt). That distinction matters downstream: a
"web-rescued" answer is not the same thing as a hallucination, and
evaluation/refusal_metrics.py breaks results out by merge_state rather
than conflating them.

Usage:
    python -m evaluation.run_eval                  # default (web enabled)
    python -m evaluation.run_eval --corpus-only     # enable_web=False
    python -m evaluation.run_eval --limit 5         # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from engine.execution_engine import RageEngine

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


def run_eval(
    golden_set: List[Dict[str, Any]],
    *,
    enable_web: bool,
) -> List[Dict[str, Any]]:
    engine = RageEngine(enable_web=enable_web)
    results: List[Dict[str, Any]] = []

    try:
        for i, record in enumerate(golden_set, start=1):
            query = record["query"]
            print(f"[{i}/{len(golden_set)}] {query}")

            try:
                execution = engine.run(query)
            except Exception as exc:
                print(f"  ⚠️ engine.run failed: {exc}")
                results.append(
                    {
                        "id": record["id"],
                        "query": query,
                        "should_refuse": record.get("should_refuse"),
                        "error": str(exc),
                    }
                )
                continue

            agent = execution["agent_decisions"]
            kpis = execution["kpis"]
            evidence = execution["evidence"]

            results.append(
                {
                    "id": record["id"],
                    "query": query,
                    "category": record.get("category"),
                    "should_refuse": record.get("should_refuse"),
                    "final_answer": execution["final_answer"],
                    "retrieved_contexts": [
                        {
                            "content": c.get("content"),
                            "source_title": c.get("source_title"),
                            "score": c.get("score"),
                            "rerank_score": c.get("rerank_score"),
                        }
                        for c in evidence
                    ],
                    "task": agent.get("task"),
                    "answer_capability": agent.get("answer_capability"),
                    "merge_state": agent.get("merge_state"),
                    "quality_status": (agent.get("quality") or {}).get("status"),
                    "quality": agent.get("quality"),
                    "quality_pre_web": agent.get("quality_pre_web"),
                    "output_validation": agent.get("output_validation"),
                    "engine_latency_ms": kpis.get("engine_latency_ms"),
                    "llm_latency_ms": kpis.get("llm_latency_ms"),
                    "prompt_tokens": kpis.get("prompt_tokens"),
                    "completion_tokens": kpis.get("completion_tokens"),
                    "cost_usd": kpis.get("cost_usd"),
                    "stage_latency_ms": {
                        k.replace("latency::", "", 1): v["avg"]
                        for k, v in execution["raw_metrics"]["distributions"].items()
                        if k.startswith("latency::")
                    },
                }
            )
    finally:
        engine.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden set through RageEngine")
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="Disable web fallback (enable_web=False) for this run",
    )
    parser.add_argument(
        "--input",
        default=str(GOLDEN_SET_PATH),
        help="Path to golden_set.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N records (smoke testing)",
    )
    args = parser.parse_args()

    golden_set = _load_golden_set(Path(args.input))
    if args.limit:
        golden_set = golden_set[: args.limit]

    results = run_eval(golden_set, enable_web=not args.corpus_only)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_corpusonly" if args.corpus_only else "_default"
    out_path = RESULTS_DIR / f"runs_{date.today().isoformat()}{suffix}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errors = sum(1 for r in results if "error" in r)
    print(f"\nWrote {len(results)} records -> {out_path}")
    if errors:
        print(f"  ⚠️ {errors} record(s) errored during engine.run()")


if __name__ == "__main__":
    main()
