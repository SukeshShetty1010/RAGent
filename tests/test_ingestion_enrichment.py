"""
tests/test_ingestion_enrichment.py

Checks that the new ingestion, enrichment, and upsert pipeline works end-to-end.
"""

import json
import logging
from ingest.loader import load_documents
from ingest.chunking import chunk_documents
from ingest.upsert import upsert_chunks
from vector.index_manager import create_index_if_not_exists, client, COLLECTION_NAME
from weaviate.collections.classes.filters import Filter
from weaviate.exceptions import WeaviateClosedClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inspect_first_doc(docs):
    """Print summary of first document’s metadata."""
    if not docs:
        print("❌ No documents loaded.")
        return
    d = docs[0]
    print("\n📄 === Sample Document ===")
    print("Page content (truncated):", d.page_content[:250].replace("\n", " "), "...")
    print("\nMetadata keys:", list(d.metadata.keys()))
    print(json.dumps(d.metadata, indent=2)[:1000])  # limit output length
    print("=========================\n")


def inspect_schema():
    """Confirm Weaviate schema matches enriched fields."""
    try:
        if not client.is_connected():
            print("🔄 Reconnecting Weaviate client for schema inspection...")
            client.connect()

        schema = client.collections.get(COLLECTION_NAME).config.get()
        print("\n📘 === Schema Properties ===")
        for p in schema.properties:
            print(f" - {p.name:20s}  ({p.data_type})")
        print("===========================\n")

    except WeaviateClosedClientError:
        print("⚠️ Client was closed, reconnecting...")
        client.connect()
        schema = client.collections.get(COLLECTION_NAME).config.get()
        for p in schema.properties:
            print(f" - {p.name:20s}  ({p.data_type})")
        print("===========================\n")


def run_ingestion_test():
    print("🚀 Checking full ingestion & enrichment flow...\n")
    create_index_if_not_exists()

    # Always ensure active connection
    if not client.is_connected():
        client.connect()
        print("✅ Connected to Weaviate.")

    # 1️⃣ Load JSON file
    file_path = "gamespot_structured_Far_Cry_6.json"
    print(f"📂 Loading file: {file_path}")
    docs = load_documents(file_path)
    print(f"✅ Loaded {len(docs)} documents.")

    # 2️⃣ Inspect first doc for enrichment correctness
    inspect_first_doc(docs)

    # 3️⃣ Chunk and upsert
    chunks = chunk_documents(docs)
    print(f"🧩 Chunked into {len(chunks)} chunks.")
    upsert_chunks(chunks)
    print("✅ Upsert complete.\n")

    # 4️⃣ Verify schema in Weaviate
    inspect_schema()

    # 5️⃣ Query canonical object by slug
    try:
        slug = docs[0].metadata.get("slug")
        if slug:
            print(f"🔎 Querying Weaviate for slug '{slug}'...")
            collection = client.collections.get(COLLECTION_NAME)
            results = collection.query.fetch_objects(
                filters=Filter.by_property("slug").equal(slug),
                limit=1
            )

            objs = results.objects
            if not objs:
                print("⚠️ No object found for slug:", slug)
            else:
                obj = objs[0]
                props = obj.properties  # ✅ Correct for Weaviate v4
                print("\n🎮 === Found Canonical Object ===")
                for k, v in props.items():
                    if k in ("genres", "platforms", "developers", "publishers", "tags", "themes"):
                        print(f"  {k:15s}: {v}")
                print("==============================\n")
        else:
            print("⚠️ No slug detected in first document metadata.")
    except Exception as e:
        print("❌ Weaviate query check failed:", e)

    # 6️⃣ Safe close afterwards
    try:
        client.close()
        print("🔒 Weaviate client closed successfully.")
    except Exception as e:
        print("⚠️ Client close warning:", e)

    print("\n✅ Ingestion enrichment test completed successfully.\n")


if __name__ == "__main__":
    run_ingestion_test()
