from vector.index_manager import client
from weaviate.classes.query import Filter

def main():
    # 1. Get the collection
    collection = client.collections.get("GameKnowledge")

    # ==========================================
    # Task A: Check Existence (Search by Slug Property)
    # ==========================================
    # The string you have is likely a "slug", not a UUID.
    test_slug = "far-cry-3-2012-2012-aa60e468"
    
    print(f"Searching for object with slug: '{test_slug}'...")
    
    # We use fetch_objects with a filter instead of fetch_object_by_id
    response = collection.query.fetch_objects(
        filters=Filter.by_property("slug").equal(test_slug),
        limit=1
    )

    if response.objects:
        obj = response.objects[0]
        print(f"✅ Found object! System UUID: {obj.uuid}")
        
        # ==========================================
        # Task B: Get Properties
        # ==========================================
        print(f"Title: {obj.properties.get('title')}")
        print(f"Genres: {obj.properties.get('genres')}")
    else:
        print(f"❌ No object found with slug '{test_slug}'")

    # ==========================================
    # Task C: Count Total Objects
    # ==========================================
    count_result = collection.aggregate.over_all(total_count=True)
    print(f"\nTotal objects in 'GameKnowledge': {count_result.total_count}")

    client.close()

if __name__ == "__main__":
    main()