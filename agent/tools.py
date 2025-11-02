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
from utils.gpu_utils import get_device  # GPU support

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Knowledge-base search (Weaviate) - GPU-powered
# ---------------------------------------------------------------------------
@tool
def search_knowledge_base(
    query: str,
    top_k: int = 5,
    source: str | None = None,
) -> List[Document]:
    """
    Search the vector store (Weaviate) for the most relevant chunks.
    Uses GPU-accelerated embeddings via vector.search.
    Returns LangChain Documents with metadata for citation.
    """
    log.info(f"Searching knowledge base on {get_device().upper()}: '{query}' (top_k={top_k}, source={source})")
    return vector_search(query=query, top_k=top_k, source=source)


# ---------------------------------------------------------------------------
# 2. Live News API
# ---------------------------------------------------------------------------
_news_tool = NewsTool()   # singleton – cheap to create

@tool
def fetch_news(
    query: str,
    limit: int = 10,
    country: str = "us",
    category: str | None = None,
) -> Dict[str, Any]:
    """
    Call the APITube news endpoint (search + headlines) and return raw JSON.
    The agent will later chunk & cite if needed.
    """
    log.info(f"Fetching news on {get_device().upper()}: '{query}' (limit={limit}, country={country}, category={category})")
    filters = {}
    if country:
        filters["country"] = country
    if category:
        filters["category"] = category

    return _news_tool.fetch_both(query=query, limit=limit, **filters)


# ---------------------------------------------------------------------------
# 3. Live IGDB API
# ---------------------------------------------------------------------------
_igdb_tool = IGDBTool()   # singleton

@tool
def search_igdb(
    query: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Call IGDB for recent games + search results.
    Returns raw JSON – the agent can cite `id`.
    """
    log.info(f"Searching IGDB on {get_device().upper()}: '{query}' (limit={limit})")
    return _igdb_tool.fetch_both(query=query, limit=limit)