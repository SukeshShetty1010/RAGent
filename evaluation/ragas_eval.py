#!/usr/bin/env python3
"""
evaluation/ragas_eval.py — RAGAS Scoring (offline only)

Scores context_precision, faithfulness, and answer_relevancy over a
run_eval.py runs artifact, restricted to the 40 answerable golden-set
records (unanswerable records have no reference answer; they're
scored separately by evaluation/refusal_metrics.py).

Never imported on the Render request path — this is an offline dev
tool (see requirements-dev.txt).

- Judge: Groq llama-3.3-70b-versatile via langchain-groq, deliberately
  NOT the llama-3.1-8b-instant generator — a model shouldn't grade its
  own output, and 8B is a weak judge.
- Embeddings: evaluation/ragas_embeddings.py's GeminiEmbeddings, so
  no torch/sentence-transformers lands locally.
- Rate limits: RunConfig(max_workers=2), scored in batches, each batch
  checkpointed to evaluation/results/.ragas_checkpoint_<date>.jsonl —
  a rerun skips already-checkpointed ids instead of restarting.

Usage:
    python -m evaluation.ragas_eval
    python -m evaluation.ragas_eval --runs evaluation/results/runs_2026-08-09_default.jsonl
    python -m evaluation.ragas_eval --limit 4   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from dotenv import load_dotenv

load_dotenv()

from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import context_precision, faithfulness, answer_relevancy
from langchain_groq import ChatGroq

from evaluation.ragas_embeddings import GeminiEmbeddings

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GOLDEN_SET_PATH = Path(__file__).resolve().parent / "data" / "golden_set.jsonl"
JUDGE_MODEL = "qwen/qwen3.6-27b"
BATCH_SIZE = 10


def _build_judge(backend: str):
    """Returns (llm, judge_model_id). Independent judge backends never
    share a checkpoint/tag -- see _run_tag's backend suffix -- so a
    single results file is never scored by two different judges."""
    if backend == "gemini":
        from evaluation.gemini_judge_llm import build_gemini_judge, JUDGE_MODEL_ID

        return build_gemini_judge(), JUDGE_MODEL_ID
    return (
        LangchainLLMWrapper(
            ChatGroq(
                model=JUDGE_MODEL,
                api_key=os.environ["GROQ_API_KEY"],
                temperature=0.0,
                max_tokens=8192,
            ),
            # Groq's API rejects n>1 ("number must be at most 1"); ragas's
            # self-consistency sampling (answer_relevancy's question
            # generation, faithfulness) defaults to n>1, so this must be
            # bypassed or every such call 400s.
            bypass_n=True,
        ),
        f"groq:{JUDGE_MODEL}",
    )


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


def _run_tag(runs_path: Path) -> str:
    m = re.search(r"runs_(\d{4}-\d{2}-\d{2}(?:_\w+)?)", runs_path.stem)
    return m.group(1) if m else runs_path.stem


def _append_checkpoint(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    scored = {}
    for row in _load_jsonl(path):
        scored[row["id"]] = row
    return scored


def _corpus_fingerprint() -> Dict[str, Any]:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        )
        try:
            return {
                "games": client.count(collection_name="Game").count,
                "editorial_chunks": client.count(
                    collection_name="EditorialChunk"
                ).count,
            }
        finally:
            client.close()
    except Exception as exc:
        return {"error": str(exc)}


def _score_batch(
    batch: List[Dict[str, Any]],
    ref_map: Dict[str, Dict[str, Any]],
    llm: LangchainLLMWrapper,
    embeddings: GeminiEmbeddings,
    run_config: RunConfig,
) -> List[Dict[str, Any]]:
    samples = [
        SingleTurnSample(
            user_input=r["query"],
            response=r["final_answer"],
            retrieved_contexts=[
                c["content"]
                for c in r.get("retrieved_contexts", [])
                if c.get("content")
            ]
            or [""],
            reference=ref_map[r["id"]]["ground_truth_answer"] or "",
        )
        for r in batch
    ]
    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        show_progress=False,
        raise_exceptions=False,
    )
    df = result.to_pandas()

    rows = []
    for i, r in enumerate(batch):
        row = {"id": r["id"], "query": r["query"]}
        for metric in ("context_precision", "faithfulness", "answer_relevancy"):
            val = df.iloc[i][metric] if metric in df.columns else None
            row[metric] = None if val is None or val != val else float(val)  # NaN-safe
        rows.append(row)
    return rows


def _split_scored_and_failed(
    rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """A row with every metric null (all three RAGAS calls raised, e.g. a
    daily token-quota exhaustion) is a failure to retry, not a scored
    zero -- checkpointing it would make a rerun skip it forever even
    after the quota resets. Only rows with at least one real score are
    considered checkpointed/done."""
    scored, failed = [], []
    for row in rows:
        if any(row.get(m) is not None for m in ("context_precision", "faithfulness", "answer_relevancy")):
            scored.append(row)
        else:
            failed.append(row)
    return scored, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a runs artifact with RAGAS")
    parser.add_argument("--runs", default=None, help="Path to a run_eval.py .jsonl artifact")
    parser.add_argument("--golden", default=str(GOLDEN_SET_PATH))
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test on first N answerable records")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--judge-backend",
        choices=["groq", "gemini"],
        default="groq",
        help="groq: llama/qwen via Groq API (daily token quota). "
        "gemini: gemini-flash-latest via Gemini's OpenAI-compat endpoint (free tier).",
    )
    args = parser.parse_args()

    runs_path = Path(args.runs) if args.runs else _latest_default_run()
    if runs_path is None or not runs_path.exists():
        raise SystemExit(
            "No runs artifact found. Run `python -m evaluation.run_eval` first, "
            "or pass --runs explicitly."
        )

    runs = _load_jsonl(runs_path)
    golden = _load_jsonl(Path(args.golden))
    ref_map = {r["id"]: r for r in golden}

    answerable = [
        r for r in runs
        if "error" not in r
        and not (ref_map.get(r["id"]) or {}).get("should_refuse", True)
        and r.get("final_answer")
        and (ref_map.get(r["id"]) or {}).get("ground_truth_answer")
    ]
    if args.limit:
        answerable = answerable[: args.limit]

    tag = _run_tag(runs_path)
    if args.judge_backend != "groq":
        tag = f"{tag}_{args.judge_backend}"
    checkpoint_path = RESULTS_DIR / f".ragas_checkpoint_{tag}.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scored = _load_checkpoint(checkpoint_path)
    pending = [r for r in answerable if r["id"] not in scored]

    print(f"{len(answerable)} answerable records, {len(scored)} already checkpointed, {len(pending)} pending")

    llm, judge_model_id = _build_judge(args.judge_backend)

    if pending:
        from utils.usage_counter import UsageCounter

        embeddings = GeminiEmbeddings()
        run_config = RunConfig(max_workers=2)

        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            print(f"Scoring batch {start}-{start + len(batch)} / {len(pending)}...")
            rows = _score_batch(batch, ref_map, llm, embeddings, run_config)
            # Approximate: 3 metrics scored per record, each a judge call.
            UsageCounter.get().record(
                args.judge_backend, "ragas_judge", count=len(batch) * 3
            )
            good_rows, failed_rows = _split_scored_and_failed(rows)
            _append_checkpoint(checkpoint_path, good_rows)
            for row in good_rows:
                scored[row["id"]] = row
            if failed_rows:
                failed_ids = [r["id"] for r in failed_rows]
                print(
                    f"  ⚠️ {len(failed_rows)} record(s) fully failed (not checkpointed, "
                    f"will retry next run): {failed_ids}"
                )

    all_scored = [scored[r["id"]] for r in answerable if r["id"] in scored]

    aggregates: Dict[str, Optional[float]] = {}
    for metric in ("context_precision", "faithfulness", "answer_relevancy"):
        vals = [row[metric] for row in all_scored if row.get(metric) is not None]
        aggregates[metric] = round(sum(vals) / len(vals), 4) if vals else None

    output = {
        "judge_model": judge_model_id,
        "embeddings_model": "Gemini gemini-embedding-001 (768-dim)",
        "runs_source": str(runs_path),
        "scored_count": len(all_scored),
        "excluded_unanswerable_count": sum(
            1 for r in golden if r.get("should_refuse")
        ),
        "unreviewed_golden_records": sum(
            1 for r in golden if not r.get("reviewed")
        ),
        "corpus_fingerprint": _corpus_fingerprint(),
        "aggregates": aggregates,
        "per_query": all_scored,
    }

    out_path = RESULTS_DIR / f"ragas_{tag}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(all_scored)} scored records -> {out_path}")
    print(json.dumps(aggregates, indent=2))

    from utils.usage_counter import UsageCounter
    UsageCounter.get().flush()


if __name__ == "__main__":
    main()
