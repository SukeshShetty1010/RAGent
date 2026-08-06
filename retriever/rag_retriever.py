"""
RAG Retriever (FULLY OBSERVABLE)

Hybrid Search (BM25 Sparse + Dense Vector) via Qdrant
with Reciprocal Rank Fusion (RRF), followed by a cross-encoder
reranking pass over the fused candidates.

Instrumentation added for:
- Embedding generation
- Vector query
- Cross-encoder reranking
- Result formatting
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Dict, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding

from utils.observability import ProfileBlock, MetricsRegistry

logger = logging.getLogger("RAG_RETRIEVER")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------------------
# BM25 sparse encoding runs locally (CPU-only, no Modal dependency).
# Dense embedding and cross-encoder reranking are resolved lazily from
# Modal on first use, not at import time — importing this module (e.g.
# for test collection) must not require Modal credentials or network
# access. Loading torch/sentence-transformers in-process instead was
# rejected: alongside fastembed's BM25 encoder it exceeded Render's
# 512MB free-tier RAM limit.

# BM25 sparse encoder (lightweight, CPU-only)
bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")

_dense_encoder_app = None
_reranker_app = None


def _get_dense_encoder():
    global _dense_encoder_app
    if _dense_encoder_app is None:
        import modal

        E5Embedder = modal.Cls.from_name("editorial-embedding-service", "E5Embedder")
        _dense_encoder_app = E5Embedder()
    return _dense_encoder_app


def _get_reranker():
    global _reranker_app
    if _reranker_app is None:
        import modal

        CrossEncoderReranker = modal.Cls.from_name(
            "cross-encoder-rerank-service", "CrossEncoderReranker"
        )
        _reranker_app = CrossEncoderReranker()
    return _reranker_app

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

    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Hybrid retrieval: BM25 Sparse + Dense Vector with RRF
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if self.client is None:
            raise RuntimeError(
                "RAGRetriever used after close() was called"
            )

        with ProfileBlock("LocalVectorSearch"):

            # --------------------------------------------------
            # Embedding generation (dense + sparse)
            # --------------------------------------------------
            with ProfileBlock("EmbeddingGeneration"):
                dense_vec = _get_dense_encoder().embed_texts.remote([query])[0]
                sparse_emb = list(bm25_encoder.query_embed(query))[0]

            MetricsRegistry.get().observe(
                "embedding_batch_size", 1
            )

            # --------------------------------------------------
            # Hybrid search (BM25 + Dense via RRF)
            # --------------------------------------------------
            fetch_limit = max(limit * 4, 20)

            with ProfileBlock("VectorQuery"):
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

            # --------------------------------------------------
            # Result formatting (RRF-ordered candidates)
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
                            # Original RRF fusion score — consumed by
                            # quality_gate.py's threshold logic. Never
                            # overwritten by the reranker below.
                            "score": point.score,
                        }
                    )

            # --------------------------------------------------
            # Cross-encoder reranking (fail-soft)
            # --------------------------------------------------
            with ProfileBlock("CrossEncoderRerank"):
                results = self._rerank(query, candidates, limit)

            MetricsRegistry.get().observe(
                "retrieval_results_count", len(results)
            )

            return results

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
        RRF ordering.
        """
        if not candidates:
            return candidates

        try:
            contents = [c.get("content") or "" for c in candidates]
            rerank_scores = _get_reranker().rerank.remote(query, contents)

            for c, s in zip(candidates, rerank_scores):
                c["rerank_score"] = float(s)

            candidates.sort(
                key=lambda c: c["rerank_score"], reverse=True
            )

        except Exception as exc:
            logger.warning(
                f"Cross-encoder rerank unavailable (fail-soft): {exc}"
            )

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
