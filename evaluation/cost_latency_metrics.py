#!/usr/bin/env python3
"""
evaluation/cost_latency_metrics.py — Latency & Cost Summary

Reads a run_eval.py output artifact (evaluation/run_eval.py must have
been re-run after the token/cost capture landed, so records carry
prompt_tokens/completion_tokens/cost_usd/stage_latency_ms) and produces
p50/p95/p99 latency, per-query cost, and stage-level latency
attribution — the numbers Phase 4's README table cites.

Usage:
    python -m evaluation.cost_latency_metrics --runs evaluation/results/runs_2026-08-13_default.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _latest_default_run() -> Optional[Path]:
    candidates = sorted(RESULTS_DIR.glob("runs_*_default.jsonl"))
    return candidates[-1] if candidates else None


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def compute_cost_latency_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in records if "error" not in r]

    engine_latencies = [r["engine_latency_ms"] for r in scored if r.get("engine_latency_ms") is not None]
    llm_latencies = [r["llm_latency_ms"] for r in scored if r.get("llm_latency_ms") is not None]
    costs = [r["cost_usd"] for r in scored if r.get("cost_usd")]
    prompt_tokens = [r["prompt_tokens"] for r in scored if r.get("prompt_tokens")]
    completion_tokens = [r["completion_tokens"] for r in scored if r.get("completion_tokens")]

    stage_totals: Dict[str, List[float]] = defaultdict(list)
    for r in scored:
        for stage, ms in (r.get("stage_latency_ms") or {}).items():
            stage_totals[stage].append(ms)

    request_total = stage_totals.get("REQUEST_TOTAL", [])
    stage_attribution = {}
    if request_total:
        total_avg = sum(request_total) / len(request_total)
        for stage, values in stage_totals.items():
            if stage == "REQUEST_TOTAL":
                continue
            stage_avg = sum(values) / len(values)
            stage_attribution[stage] = {
                "avg_ms": round(stage_avg, 2),
                "pct_of_total": round(100.0 * stage_avg / total_avg, 2) if total_avg else None,
            }

    return {
        "total_scored": len(scored),
        "total_errored": len(records) - len(scored),
        "engine_latency_ms": {
            "p50": round(_percentile(engine_latencies, 0.50), 2) if engine_latencies else None,
            "p95": round(_percentile(engine_latencies, 0.95), 2) if engine_latencies else None,
            "p99": round(_percentile(engine_latencies, 0.99), 2) if engine_latencies else None,
            "mean": round(statistics.mean(engine_latencies), 2) if engine_latencies else None,
        },
        "llm_latency_ms": {
            "p50": round(_percentile(llm_latencies, 0.50), 2) if llm_latencies else None,
            "p95": round(_percentile(llm_latencies, 0.95), 2) if llm_latencies else None,
            "mean": round(statistics.mean(llm_latencies), 2) if llm_latencies else None,
        },
        "cost_usd_per_query": {
            "mean": round(statistics.mean(costs), 8) if costs else 0.0,
            "median": round(statistics.median(costs), 8) if costs else 0.0,
            "total_for_run": round(sum(costs), 6) if costs else 0.0,
        },
        "tokens_per_query": {
            "mean_prompt_tokens": round(statistics.mean(prompt_tokens), 1) if prompt_tokens else 0.0,
            "mean_completion_tokens": round(statistics.mean(completion_tokens), 1) if completion_tokens else 0.0,
        },
        "stage_latency_attribution": dict(
            sorted(stage_attribution.items(), key=lambda kv: -(kv[1]["avg_ms"] or 0))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latency and cost from a run_eval.py artifact")
    parser.add_argument(
        "--runs",
        default=None,
        help="Path to a run_eval.py output .jsonl (defaults to the latest runs_*_default.jsonl)",
    )
    args = parser.parse_args()

    runs_path = Path(args.runs) if args.runs else _latest_default_run()
    if runs_path is None or not runs_path.exists():
        raise SystemExit(
            "No runs artifact found. Run `python -m evaluation.run_eval` first, "
            "or pass --runs explicitly."
        )

    records = _load_jsonl(runs_path)
    metrics = compute_cost_latency_metrics(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = runs_path.stem.replace("runs_", "cost_latency_", 1)
    out_path = RESULTS_DIR / f"{stem}.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Scored {runs_path.name} -> {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
