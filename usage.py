import weaviate
from weaviate import WeaviateClient
from weaviate.collections.classes.filters import Filter

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
GAME_UUID = "00404ce3-ab9d-55a3-a239-6b4722f77900"  # Far Cry 5


# ------------------------------------------------------------------
# CONNECT
# ------------------------------------------------------------------
client: WeaviateClient = weaviate.connect_to_local()

try:
    # ==============================================================
    # STAGE 1 — Canonical Game
    # ==============================================================
    game_collection = client.collections.get("Game")
    game_obj = game_collection.query.fetch_object_by_id(GAME_UUID)

    print("\n==============================")
    print(" STAGE 1: CANONICAL GAME ")
    print("==============================")

    if not game_obj:
        print(f"❌ No Game found for UUID: {GAME_UUID}")
    else:
        print("✅ Game found")
        print(game_obj.properties)

    # ==============================================================
    # STAGE 2 — Platform Specs
    # ==============================================================
    platform_collection = client.collections.get("PlatformSpec")
    platform_results = platform_collection.query.fetch_objects(
        filters=Filter.by_ref("game").by_id().equal(GAME_UUID)
    )

    print("\n==============================")
    print(" STAGE 2: PLATFORM SPECS ")
    print("==============================")

    if not platform_results.objects:
        print("⚠️ No PlatformSpec objects found.")
    else:
        print(f"✅ Found {len(platform_results.objects)} PlatformSpec objects")
        for idx, obj in enumerate(platform_results.objects, start=1):
            print(f"\n--- PlatformSpec #{idx} ---")
            print(obj.properties)

    # ==============================================================
    # STAGE 3 — IGDB METADATA
    # ==============================================================
    igdb_collection = client.collections.get("IGDB_Game")
    igdb_results = igdb_collection.query.fetch_objects(
        filters=Filter.by_ref("game").by_id().equal(GAME_UUID)
    )

    print("\n==============================")
    print(" STAGE 3: IGDB METADATA ")
    print("==============================")

    if not igdb_results.objects:
        print("⚠️ No IGDB_Game objects found.")
    else:
        print(f"✅ Found {len(igdb_results.objects)} IGDB entities")

        for idx, obj in enumerate(igdb_results.objects, start=1):
            print(f"\n--- IGDB Entity #{idx} ---")
            print({
                "igdb_id": obj.properties.get("igdb_id"),
                "entity_category": obj.properties.get("entity_category"),
                "name": obj.properties.get("name"),
                "parent_game": obj.properties.get("parent_game"),
                "version_parent": obj.properties.get("version_parent"),
            })

finally:
    client.close()
