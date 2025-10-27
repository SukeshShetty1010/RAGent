# ingest/upsert.py
import hashlib
import logging
from typing import List
import weaviate  # <-- CHANGED: Added this import for v4 filters
from langchain_core.documents import Document
from vector.embed import get_embedding_model
from vector.index_manager import client
from langchain_weaviate import WeaviateVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upsert_chunks(chunks: List[Document]):
    """
    Upserts chunks to Weaviate after checking for duplicates and embedding.
    
    Args:
        chunks (List[Document]): List of chunked documents.
    """
    embeddings = get_embedding_model()
    vectorstore = WeaviateVectorStore(
        client,
        index_name="KnowledgeBase",
        text_key="text",
        embedding=embeddings,
        attributes=["source", "chunk_id", "created_at", "article_id", "content_hash"]
    )
    
    # --- CHANGED: Get collection object once (v4 syntax) ---
    collection = client.collections.get("KnowledgeBase")
    
    to_add = []
    for chunk in chunks:
        content_hash = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
        chunk.metadata['content_hash'] = content_hash
        
        try:
            # --- CHANGED: Replaced entire v3 query block with v4 ---
            
            # This is the new v4 query syntax for checking duplicates
            response = collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("content_hash").equal(content_hash),
                limit=1
            )
            
            # The v4 response is a list of objects. If the list is empty, the doc is new.
            if not response.objects:
                to_add.append(chunk)

            # --- END OF CHANGES ---

        except Exception as e:
            logger.error(f"Query error checking for hash {content_hash}: {e}")
            continue
    
    # This part below was already correct and doesn't need to change
    if to_add:
        try:
            vectorstore.add_documents(to_add, batch_size=100)
            logger.info(f"Successfully upserted {len(to_add)} new chunks.")
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
    else:
        logger.info("No new chunks to upsert; all are duplicates.")