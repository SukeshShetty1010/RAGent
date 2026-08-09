#!/usr/bin/env python3
"""
evaluation/refusal_metrics.py — Falsifiable Refusal Metric

Pure, deterministic, no LLM. Grades a run_eval.py output artifact
against the golden set's `should_refuse` labels — external ground
truth, not the pipeline judging itself. `calculate_honesty_rate`
(tests/evaluation_metrics.py) is FULL+PARTIAL / total: it always
"succeeds" by construction because it has no external label to fail
against. This metric can fail.

`answer_capability == "insufficient"` is treated as a refusal (matches
engine/execution_engine.py's guard: INSUFFICIENT is the only capability
that skips LLM generation and returns the fixed refusal string).

Usage:
    python -m evaluation.refusal_metrics --runs evaluation/results/runs_2026-08-09_default.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
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


def compute_refusal_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in records if "error" not in r]
    errored = [r for r in records if "error" in r]

    tp = fp = fn = tn = 0
    by_merge_state: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "refused": 0, "should_refuse": 0}
    )
    false_answers: List[Dict[str, Any]] = []
    over_refusals: List[Dict[str, Any]] = []

    for r in scored:
        should_refuse = bool(r.get("should_refuse"))
        refused = r.get("answer_capability") == "insufficient"
        merge_state = r.get("merge_state") or "unknown"

        by_merge_state[merge_state]["total"] += 1
        by_merge_state[merge_state]["refused"] += int(refused)
        by_merge_state[merge_state]["should_refuse"] += int(should_refuse)

        if should_refuse and refused:
            tp += 1
        elif should_refuse and not refused:
            fn += 1
            false_answers.append({"id": r.get("id"), "query": r.get("query")})
        elif not should_refuse and refused:
            fp += 1
            over_refusals.append({"id": r.get("id"), "query": r.get("query")})
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    total_should_refuse = tp + fn
    total_answerable = fp + tn

    return {
        "total_scored": len(scored),
        "total_errored": len(errored),
        "refusal_precision": round(precision, 4),
        "refusal_recall": round(recall, 4),
        "refusal_f1": round(f1, 4),
        # Answered when it should have refused — the number that matters.
        "false_answer_rate": round(fn / total_should_refuse, 4)
        if total_should_refuse
        else 0.0,
        # Refused when evidence existed.
        "over_refusal_rate": round(fp / total_answerable, 4)
        if total_answerable
        else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "by_merge_state": dict(by_merge_state),
        "false_answers": false_answers,
        "over_refusals": over_refusals,
        "errored_ids": [r.get("id") for r in errored],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score refusal precision/recall against golden labels")
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
    metrics = compute_refusal_metrics(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = runs_path.stem.replace("runs_", "refusal_", 1)
    out_path = RESULTS_DIR / f"{stem}.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Scored {runs_path.name} -> {out_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k not in (
        "false_answers", "over_refusals", "errored_ids", "by_merge_state"
    )}, indent=2))


if __name__ == "__main__":
    main()
