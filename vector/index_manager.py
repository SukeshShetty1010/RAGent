# vector/index_manager.py
import os
import logging
from urllib.parse import urlparse
import weaviate
from weaviate.classes.config import Property, DataType, Configure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# CONNECTION SETUP
# --------------------------------------------------------------------------------

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
parsed = urlparse(WEAVIATE_URL)
host = parsed.hostname or "localhost"
port = parsed.port or 8080

try:
    # New API (Weaviate v4)
    client = weaviate.connect_to_local(port=port)
    logger.info("Connected to local Weaviate instance at %s:%s", host, port)
except Exception as e:
    logger.exception("❌ Failed to connect to Weaviate: %s", e)
    raise

COLLECTION_NAME = "GameKnowledge"

# --------------------------------------------------------------------------------
# SCHEMA CREATION
# --------------------------------------------------------------------------------

def create_index_if_not_exists():
    """
    Create or ensure the GameKnowledge collection exists with full enriched schema.
    """
    if client.collections.exists(COLLECTION_NAME):
        logger.info("Collection '%s' already exists.", COLLECTION_NAME)
        return

    logger.info("Creating collection '%s' with enriched schema...", COLLECTION_NAME)

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            # Core identifiers
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="game_id", data_type=DataType.TEXT),  # use TEXT, not INT
            Property(name="unified_game_id", data_type=DataType.TEXT),
            Property(name="slug", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),

            # Dates
            Property(name="release_date", data_type=DataType.DATE),
            Property(name="release_year", data_type=DataType.INT),
            Property(name="created_at", data_type=DataType.DATE),
            Property(name="updated_at", data_type=DataType.DATE),

            # Arrays (stored as TEXT[] in Weaviate)
            Property(name="genres", data_type=DataType.TEXT_ARRAY),
            Property(name="platforms", data_type=DataType.TEXT_ARRAY),
            Property(name="developers", data_type=DataType.TEXT_ARRAY),
            Property(name="publishers", data_type=DataType.TEXT_ARRAY),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(name="themes", data_type=DataType.TEXT_ARRAY),
            Property(name="stores", data_type=DataType.TEXT_ARRAY),

            # Numbers
            Property(name="rating", data_type=DataType.NUMBER),
            Property(name="rating_count", data_type=DataType.NUMBER),
            Property(name="user_rating", data_type=DataType.NUMBER),
            Property(name="critic_rating", data_type=DataType.NUMBER),
            Property(name="metacritic", data_type=DataType.NUMBER),
            Property(name="playtime", data_type=DataType.NUMBER),
            Property(name="articles_count", data_type=DataType.INT),
            Property(name="reviews_count", data_type=DataType.INT),

            # Other
            Property(name="esrb_rating", data_type=DataType.TEXT),
            Property(name="language", data_type=DataType.TEXT),
            Property(name="content_length", data_type=DataType.INT),
            Property(name="content_hash", data_type=DataType.TEXT),
            Property(name="site_detail_url", data_type=DataType.TEXT),
        ]
    )

    logger.info("✅ Created collection '%s' successfully.", COLLECTION_NAME)


# --------------------------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------------------------

def list_collections():
    """Return all available collections."""
    cols = client.collections.list_all()
    logger.info("Available collections: %s", cols)
    return cols


def delete_collection(name: str = COLLECTION_NAME):
    """Delete the given collection (useful for schema resets)."""
    if not client.collections.exists(name):
        logger.warning("Collection '%s' does not exist.", name)
        return
    client.collections.delete(name)
    logger.info("🧹 Deleted collection '%s'.", name)


def reconnect():
    """Reconnect to Weaviate (if connection closed)."""
    global client
    if client.is_connected():
        return client
    client = weaviate.connect_to_local(port=port)
    logger.info("🔄 Reconnected to Weaviate.")
    return client
