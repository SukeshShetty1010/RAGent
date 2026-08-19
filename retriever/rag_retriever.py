"""
RAG Retriever (FULLY OBSERVABLE)

Hybrid Search (BM25 Sparse + Dense Vector) via Qdrant
with Reciprocal Rank Fusion (RRF), followed by a reranking
pass over the fused candidates.

Reranking is provider-dispatched (RERANKER_PROVIDER): the in-process
fastembed cross-encoder, or Voyage's rerank HTTP API. See
_rerank_scores() below.

Instrumentation added for:
- Embedding generation
- Vector query
- Reranking
- Result formatting
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Dict, Optional, Literal

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from utils.observability import ProfileBlock, MetricsRegistry
from llm.gemini_client import embed_text
from retriever.reranker_provider import resolve_reranker_provider

logger = logging.getLogger("RAG_RETRIEVER")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------------------
# BM25 sparse encoding always runs locally (CPU-only, ONNX). Per
# CLAUDE.md it is loaded eagerly at module level rather than lazily: a
# failure to load must surface as a visible boot-time error, not an
# unexplained hang on the first live request. Dense embeddings call
# Gemini's API (llm/gemini_client.py) — no local model.
#
# Reranking is provider-dispatched via RERANKER_PROVIDER:
#   "local"      — in-process fastembed cross-encoder (default)
#   "cloudflare" — Workers AI bge-reranker-base (llm/cloudflare_rerank_client.py)
#   "hfspace"    — same model over HTTP on an HF Space (hf_space/)
#   "voyage"     — Voyage's rerank HTTP API (llm/voyage_client.py)
# The ONNX cross-encoder is constructed ONLY on the local path; that
# conditional is the whole point of the flag, since it is what actually
# keeps the ~300MB model out of the Render process.

# .env must be parsed before RERANKER_PROVIDER is read at import time —
# RAGRetriever.__init__ also calls load_dotenv(), but that runs far too
# late to influence the module-level provider decision below.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a hard dependency
    pass

# BM25 sparse encoder (lightweight, CPU-only)
bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")

# Resolved once at import, not per call: the eager-load decision below is
# made from it, so re-reading it later could dispatch to a local reranker
# that was never constructed.
RERANKER_PROVIDER = resolve_reranker_provider()

# Cross-encoder reranker — identical model to the one formerly hosted on
# Modal. retriever/quality_gate.py's local relevance floors are
# calibrated on this exact model's raw logits; do not swap it for a
# different reranker without re-running that calibration.
reranker: Optional[TextCrossEncoder] = None

if RERANKER_PROVIDER == "local":
    reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
elif RERANKER_PROVIDER == "cloudflare":
    # Validate credentials at boot rather than degrading every live query
    # to "rerank unavailable" — same eager-error principle as the local
    # model load. No request is made here.
    from llm.cloudflare_rerank_client import CLOUDFLARE_RERANK_MODEL, _endpoint, _env
    _endpoint()          # requires CLOUDFLARE_ACCOUNT_ID
    _env("CLOUDFLARE_API_TOKEN")
    logger.info(f"Reranker provider: cloudflare ({CLOUDFLARE_RERANK_MODEL})")
elif RERANKER_PROVIDER == "hfspace":
    # Validate config at boot rather than degrading every live query to
    # "rerank unavailable" — same eager-error principle as the local
    # model load. Reachability is NOT checked here: a sleeping Space
    # must not block startup, and the request path is fail-soft.
    from llm.hf_rerank_client import _get_base_url
    logger.info(f"Reranker provider: hfspace ({_get_base_url()})")
else:
    from llm.voyage_client import VOYAGE_RERANK_MODEL, _get_voyage_api_key
    _get_voyage_api_key()
    logger.info(f"Reranker provider: voyage ({VOYAGE_RERANK_MODEL})")

# ONNX Runtime's per-call memory scales with rerank() batch_size, not just
# candidate count — confirmed live: 20 real-length (~500-token) candidates
# at the fastembed default batch_size=64 (i.e. one batch) peaked around
# 900MB RSS and OOM-killed the Render container on the very first live
# query. batch_size=1 processes candidates sequentially instead, measured
# flat at ~305MB RSS (reranker + one full rerank call) with no measurable
# latency cost on this CPU-bound workload. Do not raise this without
# re-measuring peak RSS under Render's 512MB cap. Local path only — the
# Voyage path sends all candidates in one HTTP request.
_RERANK_BATCH_SIZE = 1


def _rerank_scores(query: str, contents: List[str]) -> List[float]:
    """Relevance scores for `contents` against `query`, in input order.

    Single dispatch point for both rerank call sites (`_rerank` and
    `score_relevance`) — they must always use the same backend, since
    quality_gate.py compares local and web `rerank_score` values on one
    scale. Raises on failure; both callers are fail-soft around it.

    NOTE the scales differ by provider: local and hfspace both return
    raw ms-marco logits (~-8..+11) from the same model, while cloudflare
    and voyage both return normalized 0..1 from different models — so
    each needs its own calibrated floors.

    One score per document is a contract, not a convention: `_rerank`
    zips the result onto candidates and `score_relevance`'s result is
    zipped onto web chunks (orchestrator.py). A short list makes `zip`
    truncate silently, corrupting quality_gate's max_relevance (a max()
    over whichever subset got scored) and producing exactly the
    mixed-scale list context_algorithms.relevance_key must never sort.
    cloudflare and hfspace already raise on a length mismatch
    themselves; this guard is what actually protects `local` (the
    default provider, a bare generator with no such check) and `voyage`
    (which can't return short but silently pads with 0.0 instead).
    """
    if not contents:
        return []

    if RERANKER_PROVIDER == "cloudflare":
        from llm.cloudflare_rerank_client import rerank as cf_rerank
        scores = cf_rerank(query, contents)

    elif RERANKER_PROVIDER == "hfspace":
        from llm.hf_rerank_client import rerank as hf_rerank
        scores = hf_rerank(query, contents)

    elif RERANKER_PROVIDER == "voyage":
        from llm.voyage_client import rerank as voyage_rerank
        scores = voyage_rerank(query, contents)

    else:
        scores = [
            float(s)
            for s in reranker.rerank(query, contents, batch_size=_RERANK_BATCH_SIZE)
        ]

    if len(scores) != len(contents):
        raise ValueError(
            f"{RERANKER_PROVIDER} reranker returned {len(scores)} scores "
            f"for {len(contents)} documents"
        )

    return scores

# ---------------------------------------------------------------------
# RAG Retriever
# ---------------------------------------------------------------------

class RAGRetriever:
    """
    Thin, stateful wrapper around Qdrant + embedding service.

    IMPORTANT:
    - Owns a Qdrant client
    - MUST be explicitly closed by caller
    """

    def __init__(self) -> None:
        self.client: Optional[QdrantClient] = None

        try:
            from dotenv import load_dotenv
            load_dotenv()

            url = os.environ.get("QDRANT_URL", "http://localhost:6333")
            api_key = os.environ.get("QDRANT_API_KEY", "")
            self.client = QdrantClient(
                url=url,
                api_key=api_key or None,
            )
        except Exception as exc:
            raise RuntimeError(
                "❌ Failed to connect to Qdrant. Is it running?"
            ) from exc

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        *,
        mode: Literal[
            "hybrid_rerank", "hybrid", "dense", "bm25"
        ] = "hybrid_rerank",
    ) -> List[Dict]:
        """
        Retrieval, defaulting to today's production path (hybrid RRF
        fusion + cross-encoder rerank).

        `mode` is ablation-only plumbing (see evaluation/ablation.py):
        - "hybrid_rerank" (default): BM25 + dense via RRF, then reranked.
          Identical to pre-`mode` behavior.
        - "hybrid": BM25 + dense via RRF, rerank skipped.
        - "dense": dense vector search only.
        - "bm25": BM25 sparse search only.
        Production call sites (retriever/orchestrator.py) never pass a
        non-default mode.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if self.client is None:
            raise RuntimeError(
                "RAGRetriever used after close() was called"
            )

        with ProfileBlock("LocalVectorSearch"):

            # --------------------------------------------------
            # Embedding generation (only what this mode needs)
            # --------------------------------------------------
            needs_dense = mode in ("hybrid_rerank", "hybrid", "dense")
            needs_sparse = mode in ("hybrid_rerank", "hybrid", "bm25")

            with ProfileBlock("EmbeddingGeneration"):
                dense_vec = (
                    embed_text(query, task_type="RETRIEVAL_QUERY")
                    if needs_dense
                    else None
                )
                sparse_emb = (
                    list(bm25_encoder.query_embed(query))[0]
                    if needs_sparse
                    else None
                )

            MetricsRegistry.get().observe(
                "embedding_batch_size", 1
            )

            # --------------------------------------------------
            # Vector search (hybrid RRF, or single-branch for ablation)
            # --------------------------------------------------
            fetch_limit = max(limit * 4, 20)

            with ProfileBlock("VectorQuery"):
                if mode in ("hybrid_rerank", "hybrid"):
                    response = self.client.query_points(
                        collection_name="EditorialChunk",
                        prefetch=[
                            models.Prefetch(
                                query=models.SparseVector(
                                    indices=sparse_emb.indices.tolist(),
                                    values=sparse_emb.values.tolist(),
                                ),
                                using="bm25",
                                limit=20,
                            ),
                            models.Prefetch(
                                query=dense_vec,
                                using="dense",
                                limit=20,
                            ),
                        ],
                        query=models.FusionQuery(
                            fusion=models.Fusion.RRF,
                        ),
                        limit=fetch_limit,
                        with_payload=[
                            "content",
                            "source_title",
                            "chunk_index",
                        ],
                    )
                elif mode == "dense":
                    response = self.client.query_points(
                        collection_name="EditorialChunk",
                        query=dense_vec,
                        using="dense",
                        limit=fetch_limit,
                        with_payload=[
                            "content",
                            "source_title",
                            "chunk_index",
                        ],
                    )
                else:  # mode == "bm25"
                    response = self.client.query_points(
                        collection_name="EditorialChunk",
                        query=models.SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist(),
                        ),
                        using="bm25",
                        limit=fetch_limit,
                        with_payload=[
                            "content",
                            "source_title",
                            "chunk_index",
                        ],
                    )

            # --------------------------------------------------
            # Result formatting (fusion/search-ordered candidates)
            # --------------------------------------------------
            with ProfileBlock("ResultFormatting"):
                candidates: List[Dict] = []

                for point in response.points:
                    candidates.append(
                        {
                            "content": point.payload.get("content"),
                            "source_title": point.payload.get(
                                "source_title"
                            ),
                            "chunk_index": point.payload.get(
                                "chunk_index"
                            ),
                            # Original fusion/search score — consumed by
                            # quality_gate.py's threshold logic. Never
                            # overwritten by the reranker below.
                            "score": point.score,
                        }
                    )

            # --------------------------------------------------
            # Cross-encoder reranking (fail-soft; skipped outside
            # the default mode)
            # --------------------------------------------------
            if mode == "hybrid_rerank":
                with ProfileBlock("CrossEncoderRerank"):
                    results = self._rerank(query, candidates, limit)
            else:
                results = candidates[:limit]

            MetricsRegistry.get().observe(
                "retrieval_results_count", len(results)
            )

            return results

    # --------------------------------------------------
    # Public relevance scoring (for evidence already retrieved
    # elsewhere, e.g. web search results — same reranker backend and
    # therefore the same scale as rerank_score, so quality_gate can
    # compare local and web evidence).
    # --------------------------------------------------

    def score_relevance(self, query: str, contents: List[str]) -> List[Optional[float]]:
        """
        Score `contents` against `query` with the same reranker backend
        used by `_rerank`. Fail-soft: on any reranker error, returns
        None for every item rather than raising, so a caller merging
        these scores into evidence never crashes on a degraded reranker.

        None, not 0.0: callers must leave `rerank_score` unset on
        failure so quality_gate.py takes its "no rerank_score" skip
        path. A sentinel 0.0 was harmless on the local logit scale but
        sits below any sane refuse floor on Voyage's normalized 0..1
        scale — i.e. an outage would have turned every web-augmented
        query into a refusal.
        """
        if not contents:
            return []

        try:
            return list(_rerank_scores(query, contents))
        except Exception as exc:
            logger.warning(
                f"Reranker unavailable (fail-soft): {exc}"
            )
            return [None] * len(contents)

    # --------------------------------------------------
    # Reranking
    # --------------------------------------------------

    def _rerank(
        self,
        query: str,
        candidates: List[Dict],
        limit: int,
    ) -> List[Dict]:
        """
        Reorder RRF candidates by cross-encoder relevance and slice
        to `limit`. Adds a `rerank_score` field; leaves the original
        `score` (RRF fusion score) untouched for downstream consumers.

        Fail-soft: on any reranker error, falls back to the original
        RRF ordering and leaves NO candidate carrying a `rerank_score`.
        Scoring and mutation are separate phases on purpose — a
        half-scored list (some candidates with `rerank_score`, some
        without) is worse than none at all, since
        context_algorithms.relevance_key falls back to comparing `score`
        across the whole list the moment even one chunk is missing it,
        silently discarding every score that WAS computed.
        """
        if not candidates:
            return candidates

        try:
            contents = [c.get("content") or "" for c in candidates]
            scored = [float(s) for s in _rerank_scores(query, contents)]
            if len(scored) != len(candidates):
                raise ValueError(
                    f"reranker returned {len(scored)} scores for "
                    f"{len(candidates)} candidates"
                )
        except Exception as exc:
            logger.warning(
                f"Reranker unavailable (fail-soft): {exc}"
            )
            return candidates[:limit]

        for c, s in zip(candidates, scored):
            c["rerank_score"] = s

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:limit]

    # --------------------------------------------------
    # Resource Management
    # --------------------------------------------------

    def close(self) -> None:
        """
        Explicit teardown of Qdrant resources.

        - Idempotent
        - Safe to call multiple times
        - REQUIRED for UI / CI / long-running processes
        """
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None


# ---------------------------------------------------------------------
# Prompt Engineering (UNCHANGED)
# ---------------------------------------------------------------------

def format_llama3_prompt(query: str, chunks: List[Dict]) -> str:
    context_blocks: List[str] = []

    for chunk in chunks:
        source = chunk.get("source_title") or "Unknown Source"
        content = (chunk.get("content") or "").strip()

        context_blocks.append(
            f"[Source: {source}]\n{content}\n"
        )

    context_str = "\n".join(context_blocks)

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful and honest assistant.\n"
        "Answer strictly based on the provided context.\n"
        "You must cite the source title for every key fact you mention using the format:\n"
        "(Source: 'Title').\n"
        "If the answer is not present in the context, say \"I don't know\".\n\n"
        "Context:\n"
        f"{context_str}"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{query}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


# ---------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Retriever CLI (Hybrid + Cited)"
    )
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )

    args = parser.parse_args()

    retriever = RAGRetriever()

    try:
        with ProfileBlock("REQUEST_TOTAL"):
            chunks = retriever.retrieve(args.query, limit=args.limit)

        if not chunks:
            print("\n⚠️ No relevant context found.\n")
            return

        print("\n================ Retrieved Context ================\n")
        for i, c in enumerate(chunks, start=1):
            print(f"[{i}] {c['source_title']} (score={c['score']:.4f})")
            print(c["content"][:500])
            print("-" * 70)

        prompt = format_llama3_prompt(args.query, chunks)

        try:
            from llm.ragent_client import chat_completion_remote
        except Exception as exc:
            print(
                "\n⚠️ Generation skipped: unable to import ragent_client\n"
            )
            print(f"Reason: {exc}")
            return

        print("\n--- Generated Answer ---\n")

        with ProfileBlock("LLMGeneration"):
            answer = chat_completion_remote(
                prompt,
                max_tokens=512,
                temperature=0.1,
            )

        print(answer.strip())
        print("\n------------------------\n")

        # --------------------------------------------------
        # Final observability report
        # --------------------------------------------------
        print("\n--- OBSERVABILITY REPORT ---")
        print(
            MetricsRegistry.get().generate_report()
        )

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
