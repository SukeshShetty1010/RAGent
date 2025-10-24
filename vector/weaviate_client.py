# vector/weaviate_client.py
import weaviate
import os
from weaviate.classes.config import Property, DataType

# Load from env or config
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
)

def ensure_schema():
    """Ensure schema for Games and News exists"""
    schema = client.schema.get()
    existing_classes = [cls["class"] for cls in schema["classes"]]

    if "Games" not in existing_classes:
        client.schema.create_class({
            "class": "Games",
            "description": "Game data from IGDB",
            "properties": [
                Property(name="title", data_type=DataType.TEXT),
                Property(name="description", data_type=DataType.TEXT),
                Property(name="genres", data_type=DataType.TEXT_ARRAY),
            ]
        })
    if "News" not in existing_classes:
        client.schema.create_class({
            "class": "News",
            "description": "Game-related news data",
            "properties": [
                Property(name="title", data_type=DataType.TEXT),
                Property(name="summary", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ]
        })

def add_data(class_name: str, properties: dict, vector: list = None):
    """Add data with optional embedding vector"""
    try:
        client.data_object.create(
            data_object=properties,
            class_name=class_name,
            vector=vector
        )
    except Exception as e:
        print(f"[ERROR] Failed to add {class_name} data: {e}")

def search(vector: list, class_name: str, top_k: int = 5):
    """Search for similar vectors"""
    try:
        result = (
            client.query
            .get(class_name, ["title", "description", "genres"])
            .with_near_vector({"vector": vector})
            .with_limit(top_k)
            .do()
        )
        return result.get("data", {}).get("Get", {}).get(class_name, [])
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return []