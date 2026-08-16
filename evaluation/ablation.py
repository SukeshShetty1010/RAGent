#!/usr/bin/env python3
"""
evaluation/ablation.py — Retrieval Ablation Study

Answers: does hybrid RRF fusion + cross-encoder reranking
(retriever/rag_retriever.py's default `mode="hybrid_rerank"`) actually
earn its complexity on this corpus, compared to dense-only, BM25-only,
and hybrid-without-rerank?

Two signals, over the golden set's 40 answerable queries:
  1. Retrieval-only precision@k / entity coverage (cheap, deterministic,
     no LLM) — all 40 queries x 4 modes.
  2. RAGAS context_precision (LLM-judged) on a 20-query subset, to keep
     judge calls bounded.

If RRF+rerank does not win, that is reported as-is — a negative result
is a real finding for the README, not a bug to hide.

Usage:
    python -m evaluation.ablation
    python -m evaluation.ablation --skip-ragas      # retrieval-only, fast
    python -m evaluation.ablation --limit 5         # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from dotenv import load_dotenv

load_dotenv()

from retriever.rag_retriever import RAGRetriever
from tests.evaluation_metrics import calculate_precision_at_k, calculate_entity_coverage

MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]
GOLDEN_SET_PATH = Path(__file__).resolve().parent / "data" / "golden_set.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAGAS_SUBSET_SIZE = 20


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def retrieval_only_ablation(
    golden_answerable: List[Dict[str, Any]], limit: int = 5
) -> Dict[str, Any]:
    retriever = RAGRetriever()
    per_mode: Dict[str, Dict[str, List[float]]] = {
        mode: {"precision_at_k": [], "entity_coverage": []} for mode in MODES
    }

    try:
        for i, record in enumerate(golden_answerable, start=1):
            query = record["query"]
            expected_titles = record.get("expected_source_titles") or []
            expected_entities = record.get("expected_entities") or []
            print(f"  [{i}/{len(golden_answerable)}] {query}")

            for mode in MODES:
                chunks = retriever.retrieve(query, limit=limit, mode=mode)
                p = calculate_precision_at_k(
                    chunks, expected_titles, k=min(limit, len(chunks)) or limit
                )
                e = calculate_entity_coverage(chunks, expected_entities)
                per_mode[mode]["precision_at_k"].append(p.precision)
                per_mode[mode]["entity_coverage"].append(e.coverage)
    finally:
        retriever.close()

    summary = {}
    for mode, vals in per_mode.items():
        n = len(vals["precision_at_k"])
        summary[mode] = {
            "mean_precision_at_k": round(sum(vals["precision_at_k"]) / n, 4) if n else 0.0,
            "mean_entity_coverage": round(sum(vals["entity_coverage"]) / n, 4) if n else 0.0,
            "n": n,
        }
    return summary


def ragas_context_precision_ablation(
    golden_subset: List[Dict[str, Any]], limit: int = 5, judge_backend: str = "groq"
) -> Dict[str, Any]:
    from ragas import evaluate, RunConfig
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import context_precision

    from evaluation.ragas_embeddings import GeminiEmbeddings
    from evaluation.ragas_eval import _build_judge

    retriever = RAGRetriever()
    llm, _judge_model_id = _build_judge(judge_backend)
    embeddings = GeminiEmbeddings()
    run_config = RunConfig(max_workers=2)

    summary: Dict[str, Any] = {}

    try:
        for mode in MODES:
            print(f"  RAGAS context_precision for mode={mode}...")
            samples = []
            for record in golden_subset:
                chunks = retriever.retrieve(record["query"], limit=limit, mode=mode)
                samples.append(
                    SingleTurnSample(
                        user_input=record["query"],
                        retrieved_contexts=[c.get("content") or "" for c in chunks] or [""],
                        reference=record.get("ground_truth_answer") or "",
                    )
                )
            dataset = EvaluationDataset(samples=samples)
            result = evaluate(
                dataset=dataset,
                metrics=[context_precision],
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
                show_progress=False,
                raise_exceptions=False,
            )
            df = result.to_pandas()
            vals = [v for v in df["context_precision"].tolist() if v == v]  # drop NaN
            summary[mode] = {
                "mean_context_precision": round(sum(vals) / len(vals), 4) if vals else None,
                "n": len(vals),
            }

            from utils.usage_counter import UsageCounter
            UsageCounter.get().record(judge_backend, "ablation", count=len(samples))
    finally:
        retriever.close()

    return summary


def _markdown_table(retrieval_summary: Dict[str, Any], ragas_summary: Dict[str, Any] | None) -> str:
    header = "| Mode | Precision@K | Entity Coverage |"
    sep = "|---|---|---|"
    if ragas_summary:
        header += " RAGAS Context Precision |"
        sep += "---|"

    lines = [header, sep]
    for mode in MODES:
        r = retrieval_summary[mode]
        row = f"| `{mode}` | {r['mean_precision_at_k']:.4f} | {r['mean_entity_coverage']:.4f} |"
        if ragas_summary:
            cp = ragas_summary.get(mode, {}).get("mean_context_precision")
            row += f" {cp:.4f} |" if cp is not None else " n/a |"
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval ablation study over the golden set")
    parser.add_argument("--golden", default=str(GOLDEN_SET_PATH))
    parser.add_argument("--skip-ragas", action="store_true", help="Skip the RAGAS context_precision pass")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test on first N answerable records")
    parser.add_argument(
        "--judge-backend",
        choices=["groq", "gemini"],
        default="groq",
        help="Same choice as ragas_eval.py -- groq (daily token quota) or gemini (gemini-flash-latest, free tier).",
    )
    args = parser.parse_args()

    golden = _load_jsonl(Path(args.golden))
    answerable = [r for r in golden if not r.get("should_refuse")]
    if args.limit:
        answerable = answerable[: args.limit]

    print(f"Retrieval-only ablation over {len(answerable)} answerable queries x {len(MODES)} modes...")
    retrieval_summary = retrieval_only_ablation(answerable)

    ragas_summary = None
    if not args.skip_ragas:
        subset = [r for r in answerable if r.get("ground_truth_answer")][:RAGAS_SUBSET_SIZE]
        print(f"RAGAS context_precision ablation over {len(subset)}-query subset x {len(MODES)} modes...")
        ragas_summary = ragas_context_precision_ablation(subset, judge_backend=args.judge_backend)

    table = _markdown_table(retrieval_summary, ragas_summary)

    baseline = retrieval_summary["hybrid_rerank"]["mean_precision_at_k"]
    best_other = max(
        retrieval_summary[m]["mean_precision_at_k"] for m in MODES if m != "hybrid_rerank"
    )
    rrf_wins = baseline >= best_other

    output = {
        "retrieval_only": retrieval_summary,
        "ragas_context_precision": ragas_summary,
        "ragas_judge_backend": args.judge_backend if not args.skip_ragas else None,
        "hybrid_rerank_wins_on_precision_at_k": rrf_wins,
        "markdown_table": table,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ablation_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote ablation results -> {out_path}")
    print()
    print(table)
    if not rrf_wins:
        print(
            "\nNOTE: hybrid_rerank did NOT win on precision@k against the best single-mode "
            "baseline on this corpus. Report this as-is."
        )

    from utils.usage_counter import UsageCounter
    UsageCounter.get().flush()


if __name__ == "__main__":
    main()
