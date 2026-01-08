"""
RAG Retriever CLI

Retrieves relevant EditorialChunk objects from Weaviate
and generates an answer using a Modal-hosted Llama 3 model.

Usage:
    python -m retriever.rag_retriever \
        --query "What are the main criticisms of Far Cry 5?" \
        --limit 5
"""

from __future__ import annotations

import argparse
from typing import List, Dict

import modal
import weaviate


# ---------------------------------------------------------------------
# Modal Embedder (MUST exactly match ingestion)
# ---------------------------------------------------------------------

E5Embedder = modal.Cls.from_name(
    "editorial-embedding-service",
    "E5Embedder",
)


# ---------------------------------------------------------------------
# RAG Retriever
# ---------------------------------------------------------------------

class RAGRetriever:
    def __init__(self) -> None:
        # ---- Fail fast on Weaviate ----
        try:
            self.client = weaviate.connect_to_local()
        except Exception as exc:
            raise RuntimeError(
                "❌ Failed to connect to Weaviate. Is it running?"
            ) from exc

        # ---- Stateful Modal embedder ----
        self.embedder = E5Embedder()

    def close(self) -> None:
        self.client.close()

    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        # --------------------------------------------------
        # ✅ EXACT embedding logic used in Stage 5 ingestion
        # --------------------------------------------------
        vector = self.embedder.embed_texts.remote([query])[0]

        try:
            collection = self.client.collections.get("EditorialChunk")
        except Exception as exc:
            raise RuntimeError(
                "❌ EditorialChunk collection not found in Weaviate"
            ) from exc

        response = collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            return_metadata=["distance"],
            return_properties=[
                "content",
                "source_title",
                "chunk_index",
            ],
        )

        results: List[Dict] = []

        for obj in response.objects:
            results.append(
                {
                    "content": obj.properties.get("content"),
                    "source_title": obj.properties.get("source_title"),
                    "chunk_index": obj.properties.get("chunk_index"),
                    "distance": obj.metadata.distance,
                }
            )

        return results


# ---------------------------------------------------------------------
# Prompt Engineering (Llama 3 Instruct)
# ---------------------------------------------------------------------

def format_llama3_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Construct a Llama 3 Instruct–compatible prompt.
    """

    context_parts: List[str] = []

    for chunk in chunks:
        source = chunk.get("source_title") or "Unknown Source"
        content = (chunk.get("content") or "").strip()

        context_parts.append(
            f"[Source: {source}]\n{content}\n"
        )

    context_str = "\n".join(context_parts)

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful and honest assistant.\n"
        "Use the provided Context to answer the user's question.\n"
        "If the answer is not present in the context, strictly state \"I don't know\".\n\n"
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
    parser = argparse.ArgumentParser(description="RAG Retriever CLI")
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
        # --------------------------------------------------
        # Step 1: Retrieve
        # --------------------------------------------------
        chunks = retriever.retrieve(args.query, limit=args.limit)

        if not chunks:
            print("\n⚠️ No relevant context found.\n")
            return

        print("\n================ Retrieved Context ================\n")
        for i, c in enumerate(chunks, start=1):
            print(f"[{i}] {c['source_title']} (distance={c['distance']:.4f})")
            print(c["content"][:500])
            print("-" * 70)

        # --------------------------------------------------
        # Step 2: Prompt Assembly
        # --------------------------------------------------
        prompt = format_llama3_prompt(args.query, chunks)

        # --------------------------------------------------
        # Step 3: Generation (Fail-soft)
        # --------------------------------------------------
        try:
            from llm.ragent_client import chat_completion_remote
        except Exception as exc:
            print(
                "\n⚠️ Generation skipped: unable to import ragent_client "
                "(is Modal installed and authenticated?)\n"
            )
            print(f"Reason: {exc}")
            return

        print("\n--- Generated Answer ---\n")

        answer = chat_completion_remote.remote(
            prompt,
            max_tokens=512,
            temperature=0.1,
        )

        print(answer.strip())
        print("\n------------------------\n")

    finally:
        retriever.close()


if __name__ == "__main__":
    main()
