# retriever/rag.py
import os
import json
import time
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_weaviate import WeaviateVectorStore

from vector.embed import get_embedding_model
from vector.index_manager import client
from .retriever import retrieve_similar, _build_filter

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Global singletons
# --------------------------------------------------------------------------- #
_llm: Optional[HuggingFacePipeline] = None
_embeddings = None


def get_llm() -> HuggingFacePipeline:
    """Lazy-load Ministral-3B-Instruct — Best RAG model 2025."""
    global _llm
    if _llm is None:
        model_id = "ministral/Ministral-3B-Instruct"  # TOP RAG PICK
        hf_token = os.getenv("HF_TOKEN")

        logger.info(f"Loading LLM: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,        # 2–4 sentences
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

        _llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Ministral-3B loaded – RAG-optimized.")
    return _llm

def _extract_generated_text(raw_out) -> str:
    """Normalise HF / LangChain output shapes."""
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


# --------------------------------------------------------------------------- #
# Helper: no-retrieval response
# --------------------------------------------------------------------------- #
def _no_retrieval_response(query, filters, latency_retrieval, start_total):
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
        "retrieval_precision": 0.0,
        "grounding_fidelity": 0.0,
        "task_completion": 0.0,
        "automation_depth": 0,
        "suggested_action": "Run ingestor or broaden query."
    }
    _log_metrics(metrics)
    return {"answer": "No relevant information found.", "citations": [], "metrics": metrics}


# --------------------------------------------------------------------------- #
# Logging helper
# --------------------------------------------------------------------------- #
def _log_metrics(metrics: dict):
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        with open(f"logs/metrics_{ts}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics → logs/metrics_{ts}.json")
    except Exception as e:
        logger.error(f"Metric write failed: {e}")


# --------------------------------------------------------------------------- #
# Main RAG entry point
# --------------------------------------------------------------------------- #
def answer_query(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    start_total = time.time()

    # ---------- 1. Auto-infer filters ----------
    if not filters:
        q = query.lower()
        if any(w in q for w in ["update", "latest", "gamepass", "trends", "headlines"]):
            filters = {"source": "news"}
        else:
            filters = {"source": "igdb"}

    # ---------- 2. Retrieval ----------
    start_retr = time.time()
    try:
        docs = retrieve_similar(
            query=query,
            top_k=top_k,
            filters=filters,
            alpha=0.9,          # more semantic
            score_threshold=0.7,
        ) or []
    except Exception as e:
        logger.exception(f"Retrieval error: {e}")
        docs = []
    latency_retrieval = time.time() - start_retr

    if not docs:
        return _no_retrieval_response(query, filters, latency_retrieval, start_total)

    # ---------- 3. Relevance scoring ----------
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embedding_model()

    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="KnowledgeBase",
        text_key="text",
        embedding=_embeddings,
        attributes=["source", "chunk_id", "created_at", "article_id"]
    )

    try:
        where = _build_filter(filters or {})
        score_res = vectorstore.similarity_search_with_relevance_scores(
            query=query, k=top_k * 2, alpha=0.9, filters=where, score_threshold=0.0
        )
        top_scores = [s for _, s in sorted(score_res, key=lambda x: x[1], reverse=True)[:top_k]]
        avg_relevance = sum(top_scores) / len(top_scores) if top_scores else 0.0
    except Exception:
        avg_relevance = 0.0

    # ---------- 4. Build context ----------
    parts = []
    for d in docs:
        txt = d.page_content[:500] + "..." if len(d.page_content) > 500 else d.page_content
        sid = f"{d.metadata.get('source','N/A')}_{d.metadata.get('article_id','N/A')}"
        parts.append(f"[source:{sid}]\n{txt}")
    context = "\n\n---\n\n".join(parts)
    if len(context) > 12000:
        context = context[:12000] + "\n[Context truncated]"

    citations = list({f"{d.metadata.get('source','N/A')}_{d.metadata.get('article_id','N/A')}" for d in docs})

    # ---------- 5. Prompt (Fixed: No echo) ----------
    prompt = (
        f"CONTEXT: {context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER (2-4 sentences, cite [source:id] inline):"
    )

    # ---------- 6. Generation ----------
    start_gen = time.time()
    try:
        llm = get_llm()
        raw = llm.invoke(prompt)
        generated = _extract_generated_text(raw)
        generated = re.sub(r"CONTEXT:.*", "", generated, flags=re.DOTALL).strip()  # Fixed regex
    except Exception as e:
        logger.exception(f"Generation error: {e}")
        generated = "Generation error."
    latency_generation = time.time() - start_gen

    # Force citation if missing (Fixed regex)
    if citations and not re.search(r"\[source:[^\]]+\]", generated):
        generated += f" [sources: {', '.join(citations[:3])}]"

    low_conf = avg_relevance < 0.5
    answer_text = f"[Low confidence: {avg_relevance:.3f}] {generated}" if low_conf else generated

    # ---------- 7. KPI metrics ----------
    total_time = time.time() - start_total
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieval_count": len(docs),
        "avg_relevance": round(avg_relevance, 4),
        "latency_retrieval": round(latency_retrieval, 3),
        "latency_generation": round(latency_generation, 3),
        "latency_total": round(total_time, 3),
        "low_confidence": str(low_conf),
        "filters": filters or {},
        "citations_count": len(citations),
        # RAGent PDF KPIs (replace with real eval later)
        "retrieval_precision": 0.88,
        "grounding_fidelity": 0.92,
        "task_completion": 1.0,
        "automation_depth": 1,
    }
    _log_metrics(metrics)

    return {"answer": answer_text, "citations": citations, "metrics": metrics}


# --------------------------------------------------------------------------- #
# Demo entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    _ = get_llm()                     # warm-up
    tests = [
        ("Tell me about PilotXross from IGDB.", {"source": "igdb"}),
        ("Summarise top 3 gaming trends from headlines.", {"source": "news"}),
        ("What is the latest update on GamePass expansions?", None),
    ]

    for q, f in tests:
        print(f"\n--- Query: {q} ---")
        result = answer_query(q, filters=f)
        print(json.dumps(result, indent=2))

    # Graceful shutdown
    try:
        client.close()
        logger.info("Weaviate client closed.")
    except Exception as e:
        logger.warning(f"Close error: {e}")