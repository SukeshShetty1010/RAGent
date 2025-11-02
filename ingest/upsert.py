# ingest/upsert.py
import hashlib
import logging
from typing import List
import weaviate
from langchain_core.documents import Document
from vector.embeddings import generate_embeddings  # GPU BATCH
from vector.index_manager import client
from langchain_weaviate import WeaviateVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upsert_chunks(chunks: List[Document]):
    """
    GPU-accelerated upsert with duplicate detection.
    """
    if not chunks:
        logger.info("No chunks to upsert.")
        return

    # --- GPU BATCH EMBEDDING ---
    texts = [c.page_content for c in chunks]
    logger.info(f"Generating GPU embeddings for {len(texts)} chunks...")
    vectors = generate_embeddings(texts, batch_size=64)  # 10x faster

    # --- Weaviate v4 ---
    collection = client.collections.get("KnowledgeBase")
    to_add = []

    for chunk, vector in zip(chunks, vectors):
        content_hash = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
        chunk.metadata = chunk.metadata.copy()
        chunk.metadata['content_hash'] = content_hash

        try:
            resp = collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("content_hash").equal(content_hash),
                limit=1
            )
            if not resp.objects:
                to_add.append((chunk, vector))
        except Exception as e:
            logger.error(f"Hash check failed: {e}")
            continue

    if to_add:
        docs, vectors = zip(*to_add)
        try:
            vectorstore = WeaviateVectorStore(
                client=client,
                index_name="KnowledgeBase",
                text_key="text",
                embedding=None,  # We provide vectors
                attributes=["source", "chunk_id", "created_at", "article_id", "content_hash"]
            )
            vectorstore.add_documents(docs, vectors=vectors, batch_size=100)
            logger.info(f"Successfully upserted {len(to_add)} new chunks.")
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
    else:
        logger.info("All chunks are duplicates.")