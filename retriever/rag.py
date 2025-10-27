# retriever/rag.py
import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_weaviate import WeaviateVectorStore

from vector.embed import get_embedding_model
from vector.index_manager import client
from .retriever import retrieve_similar, _build_filter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache
_llm = None
_embeddings = None

def get_llm() -> HuggingFacePipeline:
    """Lazy load the HuggingFace LLM only once."""
    global _llm
    if _llm is None:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")
        model_id = "instruction-pretrain/InstructLM-500M"

        logger.info(f"Loading LLM model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        # Some models require add_bos_token=True; ensure our tokenizer is configured appropriately
        if not getattr(tokenizer, "add_bos_token", False):
            try:
                tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
                tokenizer.add_bos_token = True
            except Exception:
                pass

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            dtype="auto",
            device_map="auto",
            offload_folder="./offload",
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

        _llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("LLM loaded successfully.")
    return _llm

def _extract_generated_text(raw_out) -> str:
    """Normalize generated output from different HF/LC shapes."""
    if raw_out is None:
        return ""
    if isinstance(raw_out, str):
        return raw_out.strip()
    if isinstance(raw_out, list) and raw_out:
        first = raw_out[0]
        if isinstance(first, dict):
            for k in ("generated_text", "text", "answer"):
                if k in first:
                    return str(first[k]).strip()
            return " ".join(str(v) for v in first.values()).strip()
        return str(first).strip()
    if isinstance(raw_out, dict):
        for k in ("generated_text", "text", "answer"):
            if k in raw_out:
                return str(raw_out[k]).strip()
        return " ".join(str(v) for v in raw_out.values()).strip()
    return str(raw_out).strip()

def answer_query(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 8
) -> Dict[str, Any]:
    """Retrieve relevant context and generate grounded answer."""
    start_total = time.time()
    default_score_threshold = 0.40

    # Retrieval phase
    start_retr = time.time()
    try:
        docs = retrieve_similar(
            query=query,
            top_k=top_k,
            filters=filters,
            alpha=0.75,
            score_threshold=default_score_threshold
        ) or []
    except Exception as e:
        logger.exception("Error during retrieve_similar(): %s", e)
        docs = []
    latency_retrieval = time.time() - start_retr

    if not docs:
        metrics = {
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
        }
        return {
            "answer": "No relevant information found.",
            "citations": [],
            "metrics": metrics
        }

    global _embeddings
    if _embeddings is None:
        try:
            _embeddings = get_embedding_model()
        except Exception as e:
            logger.exception("Failed to get embedding model: %s", e)
            _embeddings = None

    try:
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name="KnowledgeBase",
            text_key="text",
            embedding=_embeddings,
            attributes=["source", "chunk_id", "created_at", "article_id", "content_hash"]
        )
    except Exception as e:
        logger.exception("Failed to construct WeaviateVectorStore: %s", e)
        vectorstore = None

    where_filter = _build_filter(filters or {})
    score_results: List[Tuple[Document, float]] = []
    try:
        if vectorstore:
            score_results = vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=top_k * 2,
                alpha=0.75,
                filters=where_filter,
                score_threshold=0.0
            )
    except Exception as e:
        logger.exception("Error computing similarity scores: %s", e)

    try:
        top_scores = [s for _, s in sorted(score_results, key=lambda x: x[1], reverse=True)[:top_k]]
        avg_relevance = sum(top_scores) / len(top_scores) if top_scores else 0.0
    except Exception:
        avg_relevance = 0.0

    retrieval_count = len(docs)
    context_parts = []
    for d in docs:
        content = d.page_content[:500] + "..." if len(d.page_content) > 500 else d.page_content
        source_id = f"{d.metadata.get('source','N/A')}-{d.metadata.get('article_id','N/A')}"
        context_parts.append(f"[source:{source_id}]\n{content}")

    context_str = "\n\n---\n\n".join(context_parts)
    if len(context_str) > 12000:
        context_str = context_str[:12000] + "\n[Context truncated for token limit]"

    try:
        citations = list({f"{d.metadata.get('source','N/A')}_{d.metadata.get('article_id','N/A')}" for d in docs})
    except Exception:
        citations = []

    # Prompt creation
    prompt = (
        f"Answer the following question using only the given context.\n"
        f"If the context does not contain enough information, reply with 'No relevant information found.'\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Provide a short, natural sentence that directly answers the question based on the context above:\n"

    )

    # Generation
    start_gen = time.time()
    try:
        llm = get_llm()
        raw_out = llm.invoke(prompt)
        generated = _extract_generated_text(raw_out)
        # Post-process to trim repeated context if echo occurs
        if "Context:" in generated:
            generated = generated.split("Context:")[-1].strip()
        if "Answer:" in generated:
            generated = generated.split("Answer:")[-1].strip()
    except Exception as e:
        logger.exception("LLM generation failed: %s", e)
        generated = f"Generation failed: {e}"
    latency_generation = time.time() - start_gen

    low_confidence = avg_relevance < 0.6
    if low_confidence:
        answer_text = f"[Low-confidence results (avg relevance: {avg_relevance:.3f}); please verify] {generated}"
    else:
        answer_text = generated

    # Metrics assembly
    total_time = time.time() - start_total
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieval_count": retrieval_count,
        "avg_relevance": round(avg_relevance, 4),
        "latency_retrieval": round(latency_retrieval, 3),
        "latency_generation": round(latency_generation, 3),
        "latency_total": round(total_time, 3),
        "low_confidence": str(low_confidence),
        "filters": filters or {},
        "citations_count": len(citations),
    }

    os.makedirs("logs", exist_ok=True)
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"logs/metrics_{timestamp}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics logged to logs/metrics_{timestamp}.json")
    except Exception:
        logger.exception("Failed to write metrics file.")

    return {
        "answer": answer_text,
        "citations": citations,
        "metrics": metrics
    }

if __name__ == "__main__":
    _ = get_llm()

    tests = [
        ("Tell me about PilotXross from IGDB.", {"source": "IGDB"}),
        ("Summarise top 3 gaming trends from headlines.", {"source": "news"}),
        ("What is the latest update on GamePass expansions?", None)
    ]

    for q, f in tests:
        print(f"\n--- Query: {q} ---")
        result = answer_query(q, filters=f)
        print(json.dumps(result, indent=2))

    try:
        if hasattr(client, "is_connected") and client.is_connected():
            client.close()
            logger.info("Weaviate client closed gracefully at shutdown.")
    except Exception as e:
        logger.warning(f"Error closing Weaviate client on shutdown: {e}")

    import gc
    gc.collect()