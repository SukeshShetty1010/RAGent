# retriever/rag.py
import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, UTC
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter

from vector.embed import get_embedding_model
from vector.index_manager import client
from .retriever import retrieve_similar, _build_filter  # Reuse retriever logic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global LLM (lazy init)
_llm = None

def get_llm() -> HuggingFacePipeline:
    global _llm
    if _llm is None:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")
        model_id = "instruction-pretrain/InstructLM-500M"
        
        logger.info(f"Loading LLM model: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype="auto",  # Replace with dtype="auto" for future-proofing
            device_map="auto",   # Allows offloading to disk
            offload_folder="./offload",  # Specify a folder for offloaded weights
            trust_remote_code=True
            )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        _llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("LLM loaded successfully.")
    
    return _llm

def answer_query(
    query: str, 
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Full RAG pipeline: Retrieve -> Generate grounded answer with citations.
    
    Returns:
        {
            "answer": str,
            "citations": List[str]  # e.g., ["IGDB_123", "news_456"]
        }
    """
    start_total = time.time()
    
    # Retrieve relevant docs
    docs = retrieve_similar(
        query=query,
        top_k=top_k,
        filters=filters,
        alpha=0.75,
        score_threshold=0.65
    )
    
    retr_time = time.time() - start_total
    
    if not docs:
        answer = "No relevant information found."
        citations = []
        avg_relevance = 0.0
        retr_count = 0
        low_conf = True
    else:
        # Compute metrics (re-run search with scores)
        embeddings = get_embedding_model()
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name="KnowledgeBase",
            text_key="text",
            embedding=embeddings,
            attributes=["source", "chunk_id", "created_at", "article_id", "content_hash"]
        )
        where_filter = _build_filter(filters or {})
        
        score_results = vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=top_k * 2,
            alpha=0.75,
            filters=where_filter,
            score_threshold=0.0  # Get all to compute avg of top
        )
        
        # Top scores avg (higher better, 0-1)
        top_scores = [score for _, score in sorted(score_results, key=lambda x: x[1], reverse=True)[:top_k]]
        avg_relevance = sum(top_scores) / len(top_scores) if top_scores else 0.0
        retr_count = len(docs)
        
        # Build context (truncate per doc + total)
        context_parts = []
        for d in docs:
            content = d.page_content[:500] + "..." if len(d.page_content) > 500 else d.page_content
            source_id = f"{d.metadata.get('source', 'N/A')}-{d.metadata.get('article_id', 'N/A')}"
            context_parts.append(f"[source:{source_id}]\n{content}")
        
        context_str = "\n\n---\n\n".join(context_parts)
        if len(context_str) > 12000:
            context_str = context_str[:12000] + "\n[Context truncated for token limit]"
        
        # Citations from retrieved docs
        citations = list(set(f"{d.metadata['source']}_{d.metadata['article_id']}" for d in docs))
        
        # System prompt
        system_prompt = """You are an AI assistant grounded in factual retrieval. Use ONLY the context provided below.
Cite sources inline like [source:article_id] after relevant facts/claims.
If information is missing or unclear, say "No relevant information found." Do not speculate."""
        
        # Llama-2-chat prompt format
        full_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

Context:
{context_str}

Question: {query}

Answer: [/INST]"""
        
        # Generate
        llm = get_llm()
        full_response = llm.invoke(full_prompt)
        
        # Extract generated answer (after [/INST])
        generated = full_response[len(full_prompt):].strip()
        if not generated:
            generated = full_response.split("Answer:")[-1].strip() if "Answer:" in full_response else "Generation failed."
        
        answer = generated
        
        # Edge cases
        low_conf = avg_relevance < 0.6
        if low_conf:
            answer = f"[Low-confidence results (avg relevance: {avg_relevance:.3f}); please verify] {answer}"
    
    gen_time = time.time() - start_total - retr_time
    total_time = time.time() - start_total
    
    # Metrics log
    os.makedirs("logs", exist_ok=True)
    metrics = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query,
        "retrieval_count": retr_count,
        "avg_relevance": round(avg_relevance, 4),
        "latency_retrieval": round(retr_time, 3),
        "latency_generation": round(gen_time, 3),
        "latency_total": round(total_time, 3),
        "low_confidence": str(low_conf),
        "filters": filters or {},
        "citations_count": len(citations)
    }
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"logs/metrics_{timestamp}.json"
    with open(log_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics logged to {log_file}")
    
    return {
        "answer": answer,
        "citations": citations,
        "metrics": metrics  # Optional, for debugging
    }

if __name__ == "__main__":
    # Test queries
    tests = [
        ("Tell me about PilotXross from IGDB.", {"source": "IGDB"}),
        ("Summarise top 3 gaming trends from headlines.", {"source": "news"}),
        ("What is the latest update on GamePass expansions?", None)
    ]
    
    for q, f in tests:
        print(f"\n--- Query: {q} ---")
        result = answer_query(q, filters=f)
        print(json.dumps(result, indent=2))