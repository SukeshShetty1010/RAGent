# retriever/retriever.py
from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter
from vector.embed import get_embedding_model
from vector.index_manager import client
import os

# Default constants (safe for CPU mode)
DEFAULT_TOP_K = 5
NEWS_SCORE_THRESHOLD = 0.2
IGDB_SCORE_THRESHOLD = 0.3      # relaxed for CPU embeddings
HYBRID_ALPHA = 0.75


def retrieve_similar(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: Optional[Dict[str, Any]] = None,
    alpha: float = HYBRID_ALPHA,
    score_threshold: float = IGDB_SCORE_THRESHOLD
) -> List[Document]:
    """
    Perform a hybrid similarity search against the GameKnowledge index.
    Compatible with CPU embeddings.
    """
    embeddings = get_embedding_model()
    device = os.getenv("DEVICE", "cpu")
    print(f"[retriever.py] Embeddings running on: {device.upper()}")

    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="GameKnowledge",   # ✅ Updated name
        text_key="text",
        embedding=embeddings,
        attributes=[
            "source", "game_id", "title", "description", "release_date",
            "created_at", "chunk_id", "genres", "platforms", "developers",
            "publishers", "tags", "themes", "franchise", "rating", "metacritic",
            "esrb_rating", "playtime", "articles_count", "reviews_count", "stores"
        ]
    )

    where_filter = _build_filter(filters or {})

    # Adjust threshold dynamically
    if filters and filters.get("source") == "news":
        score_threshold = NEWS_SCORE_THRESHOLD
    else:
        score_threshold = IGDB_SCORE_THRESHOLD

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
    """Construct a Weaviate query filter object."""
    sub_filters = []

    if "source" in filters and filters["source"]:
        sub_filters.append(Filter.by_property("source").equal(filters["source"]))

    if "game_id" in filters and filters["game_id"] is not None:
        sub_filters.append(Filter.by_property("game_id").equal(int(filters["game_id"])))

    if "created_at" in filters and isinstance(filters["created_at"], dict):
        if "gte" in filters["created_at"]:
            sub_filters.append(Filter.by_property("created_at")
                               .greater_than_or_equal(filters["created_at"]["gte"]))
        if "lte" in filters["created_at"]:
            sub_filters.append(Filter.by_property("created_at")
                               .less_than_or_equal(filters["created_at"]["lte"]))

    if not sub_filters:
        return None

    return Filter.and_(*sub_filters) if len(sub_filters) > 1 else sub_filters[0]
