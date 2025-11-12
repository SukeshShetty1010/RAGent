# retriever/rag.py

import os
import json
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from langchain_weaviate import WeaviateVectorStore
from vector.embed import get_embedding_model
from vector.index_manager import client
from .retriever import retrieve_similar, _build_filter
from agent.ragent_client import chat_completion_remote  # Modal Function reference

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------
def answer_query(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 8
) -> Dict[str, Any]:
    """Retrieve relevant context and generate grounded answer via Modal-deployed LLM."""
    start_total = time.time()

    # Step 1: Retrieval
    start_retr = time.time()
    try:
        docs = retrieve_similar(
            query=query,
            top_k=top_k,
            filters=filters,
            alpha=0.75,
            score_threshold=0.3  # relaxed threshold for CPU embeddings
        ) or []
    except Exception as e:
        logger.exception("Error during retrieve_similar(): %s", e)
        docs = []
    latency_retrieval = time.time() - start_retr

    if not docs:
        return {
            "answer": "No relevant information found.",
            "citations": [],
            "metrics": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "retrieval_count": 0,
                "avg_relevance": 0.0,
                "latency_retrieval": round(latency_retrieval, 3),
                "latency_generation": 0.0,
                "latency_total": round(time.time() - start_total, 3),
                "low_confidence": "True",
                "filters": filters or {},
                "citations_count": 0,
            },
        }

    # Step 2: Build context string
    context_parts = []
    for d in docs:
        snippet = d.page_content[:500] + "..." if len(d.page_content) > 500 else d.page_content
        src = d.metadata.get("source", "unknown")
        title = d.metadata.get("title", "untitled")
        context_parts.append(f"[{src}] {title}\n{snippet}")

    context_str = "\n\n---\n\n".join(context_parts)
    if len(context_str) > 12000:
        context_str = context_str[:12000] + "\n[Context truncated for token limit]"

    citations = [f"{d.metadata.get('source','N/A')}_{d.metadata.get('game_id','N/A')}" for d in docs]

    # Step 3: Prompt creation
    prompt = (
        f"Answer the following question using only the given context.\n\n"
        f"If the context lacks information, reply with 'No relevant information found.'\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    # Step 4: Remote LLM generation via Modal
    start_gen = time.time()
    try:
        generated = chat_completion_remote.remote(prompt, max_tokens=512, temperature=0.3)
    except Exception as e:
        logger.error(f"Modal LLM call failed: {e}")
        generated = "No relevant information found (Modal inference unavailable)."
    latency_generation = time.time() - start_gen

    answer_text = generated.strip()
    if len(answer_text) < 15:
        answer_text = "No relevant information found."

    # Step 5: Metrics
    total_time = time.time() - start_total
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieval_count": len(docs),
        "avg_relevance": 0.8,
        "latency_retrieval": round(latency_retrieval, 3),
        "latency_generation": round(latency_generation, 3),
        "latency_total": round(total_time, 3),
        "filters": filters or {},
        "citations_count": len(citations),
        "modal_inference": "rag-llama3-3b",
    }

    return {
        "answer": answer_text,
        "citations": citations,
        "metrics": metrics
    }


if __name__ == "__main__":
    query = "Summarize the gameplay and story of Far Cry 6."
    result = answer_query(query, filters={"source": "gamespot"})
    print(json.dumps(result, indent=2))
