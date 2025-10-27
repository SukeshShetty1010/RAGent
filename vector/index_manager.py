# vector/index_manager.py
import weaviate
import os
from urllib.parse import urlparse

# Import new v4 config classes
from weaviate.classes.config import (
    Configure,
    Property,
    DataType
)

# Use os.getenv for flexibility, default to your local URL
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# --- v4 Connection ---
# Parse host and port from the URL
parsed_url = urlparse(WEAVIATE_URL)
host = parsed_url.hostname or "localhost"
port = parsed_url.port or 8080

# Use the v4 connection method
client = weaviate.connect_to_local(
    host=host,
    port=port
)
# --- End of v4 Connection ---


def create_index_if_not_exists():
    """
    Creates the Weaviate collection (KnowledgeBase) if it doesn't exist.
    (Updated for Weaviate v4)
    """
    collection_name = "KnowledgeBase"
    
    # v4 check:
    if not client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' not found. Creating...")
        
        # v4 schema creation:
        client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT), # v4 uses TEXT for 'string'
                Property(name="chunk_id", data_type=DataType.INT),
                Property(name="created_at", data_type=DataType.DATE),
                Property(name="article_id", data_type=DataType.INT),
                Property(name="content_hash", data_type=DataType.TEXT), # v4 uses TEXT for 'string'
            ]
        )
        print(f"Successfully created collection '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' already exists.")