# retriever/retriever.py
from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter
from vector.embed import get_embedding_model
from vector.index_manager import client
from utils.gpu_utils import get_device
from agent.constants import (
    SOURCE_NEWS, SOURCE_IGDB,
    DEFAULT_TOP_K,
    NEWS_SCORE_THRESHOLD,
    IGDB_SCORE_THRESHOLD,
    HYBRID_ALPHA
)

def retrieve_similar(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: Optional[Dict[str, Any]] = None,
    alpha: float = HYBRID_ALPHA,
    score_threshold: float = IGDB_SCORE_THRESHOLD
) -> List[Document]:
    """
    Hybrid search with smart threshold tuning.
    """
    embeddings = get_embedding_model()
    device = get_device()

    # Debug GPU
    try:
        model_device = next(embeddings.client.model.parameters()).device
        print(f"[retriever.py] Embeddings on {model_device} ({device.upper()})")
    except:
        pass

    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="KnowledgeBase",
        text_key="text",
        embedding=embeddings,
        attributes=["source", "chunk_id", "created_at", "article_id", "content_hash"]
    )

    where_filter = _build_filter(filters or {})

    # Smart threshold: news is noisier → lower bar
    if filters and filters.get("source") == SOURCE_NEWS:
        score_threshold = NEWS_SCORE_THRESHOLD  # 0.2
    else:
        score_threshold = IGDB_SCORE_THRESHOLD  # 0.7

    try:
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=top_k * 3,
            alpha=alpha,
            filters=where_filter,
            score_threshold=score_threshold
        )

        sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
        docs = [doc for doc, score in sorted_results[:top_k]]

        return docs

    except Exception as e:
        print(f"[Retriever] Error during search: {e}")
        return []


def _build_filter(filters: Dict[str, Any]) -> Optional[Filter]:
    sub_filters = []

    source = filters.get("source")
    if source:
        # Normalize just in case
        if source in ("news", SOURCE_NEWS):
            source = SOURCE_NEWS
        elif source in ("igdb", "IGDB", SOURCE_IGDB):
            source = SOURCE_IGDB
        sub_filters.append(Filter.by_property("source").equal(source))

    article_id = filters.get("article_id")
    if article_id is not None:
        sub_filters.append(Filter.by_property("article_id").equal(int(article_id)))

    created_at = filters.get("created_at")
    if isinstance(created_at, dict):
        if "gte" in created_at:
            sub_filters.append(Filter.by_property("created_at").greater_than_or_equal(created_at["gte"]))
        if "lte" in created_at:
            sub_filters.append(Filter.by_property("created_at").less_than_or_equal(created_at["lte"]))

    if not sub_filters:
        return None
    return Filter.and_(*sub_filters) if len(sub_filters) > 1 else sub_filters[0]