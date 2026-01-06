import json
import sys
from pathlib import Path

import weaviate
from weaviate import WeaviateClient
from weaviate.exceptions import WeaviateBaseError


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

SCHEMA_DIR = Path("vector/schemas")

# Ordered by dependency (DO NOT CHANGE ORDER)
SCHEMA_FILES = [
    "rawg_game_schema.json",          # Game (anchor)
    "PlatformSpec_Schema.json",       # PlatformSpec → Game
    "IGDB_Schema.json",               # IGDB_Game → Game
    "GameSpot_Schema.JSON",           # GameSpot_Game → Game
    "editorial_chunk_schema.json",    # EditorialChunk → Game + GameSpot_Game
]


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def load_schema(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_schema_if_missing(client: WeaviateClient, schema: dict) -> None:
    class_name = schema.get("class")
    if not class_name:
        raise ValueError("Schema JSON missing 'class' field")

    existing = client.collections.list_all()

    if class_name in existing:
        print(f"✅ Schema already exists: {class_name}")
        return

    print(f"🛠️  Creating schema: {class_name}")
    client.collections.create_from_dict(schema)
    print(f"✅ Created schema: {class_name}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main() -> None:
    print("🔗 Connecting to Weaviate...")
    client: WeaviateClient = weaviate.connect_to_local()

    try:
        for filename in SCHEMA_FILES:
            schema_path = SCHEMA_DIR / filename
            schema = load_schema(schema_path)
            create_schema_if_missing(client, schema)

        print("\n🎉 All schemas are present and ready.")

    except (WeaviateBaseError, Exception) as exc:
        print(f"\n❌ Schema creation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    finally:
        client.close()
        print("🔒 Weaviate connection closed.")


# ------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
