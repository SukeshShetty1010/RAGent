import weaviate
from weaviate import WeaviateClient
from weaviate.collections.classes.filters import Filter

GAME_UUID = "00404ce3-ab9d-55a3-a239-6b4722f77900"

client: WeaviateClient = weaviate.connect_to_local()

try:
    # Stage 1: inspect Game
    game_collection = client.collections.get("Game")
    game_obj = game_collection.query.fetch_object_by_id(GAME_UUID)
    print("\n=== STAGE 1: GAME OBJECT ===")
    print(game_obj.properties if game_obj else f"❌ No Game for UUID: {GAME_UUID}")

    # Stage 2: inspect PlatformSpec with correct v4 filter
    platform_collection = client.collections.get("PlatformSpec")
    platform_results = platform_collection.query.fetch_objects(
        filters=Filter.by_ref("game").by_id().equal(GAME_UUID)
    )

    print("\n=== STAGE 2: PLATFORM SPECS ===")
    if not platform_results.objects:
        print("⚠️ No PlatformSpec objects found.")
    else:
        for idx, obj in enumerate(platform_results.objects, start=1):
            print(f"\n--- PlatformSpec #{idx} ---")
            print(obj.properties)

finally:
    client.close()
