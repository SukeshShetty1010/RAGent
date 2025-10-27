# retriever/retriever.py
from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter
from vector.embed import get_embedding_model
from vector.index_manager import client

def retrieve_similar(
    query: str, 
    top_k: int = 5, 
    filters: Optional[Dict[str, Any]] = None, 
    alpha: float = 0.75, 
    score_threshold: float = 0.7
) -> List[Document]:
    """
    Retrieve similar documents using Weaviate hybrid search (vector + BM25).
    
    Args:
        query: Search query.
        top_k: Max number of documents to return.
        filters: Dict of filters e.g., {"source": "IGDB", "created_at": {"gte": "2025-01-01"}}
        alpha: Hybrid weight (1.0 = vector only, 0.0 = keyword only).
        score_threshold: Min relevance score (0-1, higher better) to include.
    
    Returns:
        List of relevant Documents (filtered and ranked).
    """
    embeddings = get_embedding_model()
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="KnowledgeBase",
        text_key="text",
        embedding=embeddings,
        attributes=["source", "chunk_id", "created_at", "article_id", "content_hash"]
    )
    
    where_filter = _build_filter(filters or {})
    
    # Fetch more candidates to account for thresholding
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=top_k * 3,
        alpha=alpha,
        filters=where_filter,
        score_threshold=score_threshold  # Auto-filters >= threshold
    )
    
    # Take top_k highest scoring
    docs = [doc for doc, score in sorted(results_with_scores, key=lambda x: x[1], reverse=True)][:top_k]
    
    return docs

def _build_filter(filters: Dict[str, Any]) -> Optional[Filter]:
    sub_filters = []
    
    # Source filter (string equal)
    source = filters.get("source")
    if source:
        sub_filters.append(Filter.by_property("source").equal(source))
    
    # Article ID filter (int equal)
    article_id = filters.get("article_id")
    if article_id is not None:
        sub_filters.append(Filter.by_property("article_id").equal(article_id))
    
    # Created at filter (date range)
    created_at = filters.get("created_at")
    if isinstance(created_at, dict):
        if "gte" in created_at:
            sub_filters.append(Filter.by_property("created_at").greater_than_or_equal(created_at["gte"]))
        if "lte" in created_at:
            sub_filters.append(Filter.by_property("created_at").less_than_or_equal(created_at["lte"]))
    
    if not sub_filters:
        return None
    
    if len(sub_filters) == 1:
        return sub_filters[0]
    
    return Filter.and_(*sub_filters)