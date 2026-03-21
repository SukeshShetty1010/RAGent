"""
RAG Retriever (FULLY OBSERVABLE)

Hybrid Search (BM25 Sparse + Dense Vector) via Qdrant
with Reciprocal Rank Fusion (RRF).

Instrumentation added for:
- Embedding generation
- Vector query
- Result formatting
"""

from __future__ import annotations

import argparse
import os
from typing import List, Dict, Optional

import modal
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

from tests.observability import ProfileBlock, MetricsRegistry


# ---------------------------------------------------------------------
# Modal Embedder (MUST exactly match ingestion)
# ---------------------------------------------------------------------

E5Embedder = modal.Cls.from_name(
    "editorial-embedding-service",
    "E5Embedder",
)

# BM25 sparse encoder (lightweight, CPU-only)
bm25_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")


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

        self.embedder = E5Embedder()

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
                dense_vec = self.embedder.embed_texts.remote([query])[0]
                sparse_emb = list(bm25_encoder.query_embed(query))[0]

            MetricsRegistry.get().observe(
                "embedding_batch_size", 1
            )

            # --------------------------------------------------
            # Hybrid search (BM25 + Dense via RRF)
            # --------------------------------------------------
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
                    limit=limit,
                    with_payload=[
                        "content",
                        "source_title",
                        "chunk_index",
                    ],
                )

            # --------------------------------------------------
            # Result formatting
            # --------------------------------------------------
            with ProfileBlock("ResultFormatting"):
                results: List[Dict] = []

                for point in response.points:
                    results.append(
                        {
                            "content": point.payload.get("content"),
                            "source_title": point.payload.get(
                                "source_title"
                            ),
                            "chunk_index": point.payload.get(
                                "chunk_index"
                            ),
                            "score": point.score,
                        }
                    )

            MetricsRegistry.get().observe(
                "retrieval_results_count", len(results)
            )

            return results

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
            from llm.ragent_client_streaming import chat_completion_remote
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
