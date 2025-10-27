# vector/search.py
from langchain_weaviate import WeaviateVectorStore
from .embed import get_embedding_model
from .index_manager import client  # Using the v4 WeaviateClient
from typing import Optional
from weaviate.classes.query import Filter  # <-- This is the correct v4 import

def search(query: str, top_k: int = 5, source: Optional[str] = None):
    """
    Performs a similarity_search on the vector store, with optional source filter.
    
    Args:
        query (str): The query string.
        top_k (int): Number of results to return.
        source (Optional[str]): Filter results by source (e.g., "IGDB" for games).
    
    Returns:
        List[Document]: List of matching documents.
    """
    embeddings = get_embedding_model()
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="KnowledgeBase",
        text_key="text",
        embedding=embeddings,
        attributes=["source", "chunk_id", "created_at", "article_id"]
    )
    
    # --- Start of Corrected Code ---

    # 1. Create the v4 filter object (if a source is provided)
    where_filter = None
    if source:
        where_filter = Filter.by_property("source").equal(source)
    
    # 2. Perform the search.
    # The `filters` argument replaces the old `query_config`.
    # Hybrid search is often the default if your index is configured for it.
    results = vectorstore.similarity_search(
        query=query, 
        k=top_k, 
        filters=where_filter  # <-- Pass the v4 filter object here
    )

    # --- End of Corrected Code ---

    # NOTE: I have removed `client.close()`
    # You should not close the client inside the search function,
    # as it will prevent any future searches from working.
    # The client connection should be managed by your main application.
    
    return results