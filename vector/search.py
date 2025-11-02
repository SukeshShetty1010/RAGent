# vector/search.py
from langchain_weaviate import WeaviateVectorStore
from .embed import get_embedding_model
from .index_manager import client
from typing import Optional
from weaviate.classes.query import Filter

def search(query: str, top_k: int = 5, source: Optional[str] = None):
    """
    Performs GPU-accelerated similarity search with optional source filter.
    """
    embeddings = get_embedding_model()
    
    # DEBUG: Confirm GPU is used
    try:
        model_device = next(embeddings.client.model.parameters()).device
        print(f"[search.py] Embedding model running on: {model_device}")
    except:
        pass  # Not all models expose .parameters()

    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="KnowledgeBase",
        text_key="text",
        embedding=embeddings,
        attributes=["source", "chunk_id", "created_at", "article_id"]
    )
    
    where_filter = None
    if source:
        where_filter = Filter.by_property("source").equal(source)
    
    results = vectorstore.similarity_search(
        query=query,
        k=top_k,
        filters=where_filter
    )
    
    return results