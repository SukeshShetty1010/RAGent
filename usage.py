import weaviate
from weaviate import WeaviateClient
from weaviate.collections.classes.filters import Filter

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
GAME_UUID = "00404ce3-ab9d-55a3-a239-6b4722f77900"  # Far Cry 5
PREVIEW_CHUNKS = 3  # how many editorial chunks to preview


# ------------------------------------------------------------------
# CONNECT
# ------------------------------------------------------------------
client: WeaviateClient = weaviate.connect_to_local()

try:
    # ==============================================================
    # STAGE 1 — CANONICAL GAME
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
    # STAGE 2 — PLATFORM SPECS
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

    # ==============================================================
    # STAGE 4 — GAMESPOT EDITORIAL CONTAINER
    # ==============================================================
    gamespot_collection = client.collections.get("GameSpot_Game")
    gamespot_results = gamespot_collection.query.fetch_objects(
        filters=Filter.by_ref("game").by_id().equal(GAME_UUID)
    )

    print("\n==============================")
    print(" STAGE 4: GAMESPOT CONTAINER ")
    print("==============================")

    if not gamespot_results.objects:
        print("⚠️ No GameSpot_Game container found.")
        parent_editorial_uuid = None
    else:
        parent_editorial_uuid = gamespot_results.objects[0].uuid
        print("✅ GameSpot container found")
        print(f"Container UUID: {parent_editorial_uuid}")

    # ==============================================================
    # STAGE 5 — EDITORIAL CHUNKS (VECTORS)
    # ==============================================================
    editorial_collection = client.collections.get("EditorialChunk")

    editorial_results = editorial_collection.query.fetch_objects(
        filters=Filter.by_ref("game").by_id().equal(GAME_UUID),
        limit=1000,
        include_vector=True,
    )

    print("\n==============================")
    print(" STAGE 5: EDITORIAL CHUNKS ")
    print("==============================")

    if not editorial_results.objects:
        print("❌ No EditorialChunk objects found.")
    else:
        print(f"✅ Found {len(editorial_results.objects)} editorial chunks")

        # --- vector inspection ---
        sample_vector = editorial_results.objects[0].vector
        if sample_vector:
            print(f"✅ Vector present (dim={len(sample_vector)})")
        else:
            print("❌ Vector missing!")

        # --- reference validation ---
        with_game_ref = 0
        with_parent_editorial_ref = 0

        for obj in editorial_results.objects:
            refs = obj.references or {}
            if "game" in refs:
                with_game_ref += 1
            if "parent_editorial" in refs:
                with_parent_editorial_ref += 1

        print(f"🔗 game reference count: {with_game_ref}")
        print(f"🔗 parent_editorial reference count: {with_parent_editorial_ref}")

        # --- preview chunks ---
        print("\n📄 Sample Editorial Chunks:")
        for idx, obj in enumerate(editorial_results.objects[:PREVIEW_CHUNKS], start=1):
            props = obj.properties
            print(f"\n--- Chunk #{idx} ---")
            print({
                "uuid": obj.uuid,
                "chunk_index": props.get("chunk_index"),
                "content_type": props.get("content_type"),
                "content_preview": (props.get("content") or "")[:200] + "...",
            })

finally:
    client.close()
