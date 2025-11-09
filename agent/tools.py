# agent/tools.py
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.documents import Document
from data.news import NewsTool
from data.igdb import IGDBTool
from vector.search import search as vector_search
from vector.index_manager import client as weaviate_client
from weaviate.classes.query import Filter
import logging
from utils.gpu_utils import get_device
from agent.constants import SOURCE_NEWS, SOURCE_IGDB, DEFAULT_TOP_K  # ← NEW

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Knowledge-base search (Weaviate) - GPU-powered
# ---------------------------------------------------------------------------
@tool
def search_knowledge_base(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    source: str | None = None,
) -> List[Document]:
    """
    Search the vector store (Weaviate) for the most relevant chunks.
    
    Args:
        query: Search query
        top_k: Number of results (default: 5)
        source: Filter by "news" or "igdb" (uses constants.SOURCE_NEWS / SOURCE_IGDB)

    Returns:
        List[Document] with full metadata for citations.
    """
    device = get_device()
    log.info(f"Searching KB on {device.upper()}: '{query}' | top_k={top_k} | source={source}")
    
    # Map human-friendly input to exact constant
    if source == "news":
        source = SOURCE_NEWS
    elif source == "igdb" or source == "IGDB":
        source = SOURCE_IGDB

    return vector_search(query=query, top_k=top_k, source=source)


# ---------------------------------------------------------------------------
# 2. Live News API
# ---------------------------------------------------------------------------
_news_tool = NewsTool()

@tool
def fetch_news(
    query: str,
    limit: int = 10,
    country: str = "us",
    category: str | None = None,
) -> Dict[str, Any]:
    """
    Fetch fresh gaming news via APITube.
    """
    log.info(f"Fetching news on {get_device().upper()}: '{query}' (limit={limit})")
    filters = {}
    if country:
        filters["country"] = country
    if category:
        filters["category"] = category

    return _news_tool.fetch_both(query=query, limit=limit, **filters)


# ---------------------------------------------------------------------------
# 3. Live IGDB API
# ---------------------------------------------------------------------------
_igdb_tool = IGDBTool()

@tool
def search_igdb(
    query: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search IGDB for recent + matching games.
    """
    log.info(f"Searching IGDB on {get_device().upper()}: '{query}' (limit={limit})")
    return _igdb_tool.fetch_both(query=query, limit=limit)