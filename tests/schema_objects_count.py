# tests/schema_objects_count.py
import os, json, requests
WEAVIATE = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# Query all objects (paginated). We'll ask the first 100.
r = requests.get(WEAVIATE.rstrip('/') + "/v1/objects?limit=100", timeout=30)
print("status:", r.status_code)
try:
    j = r.json()
    # Print count and class samples
    objs = j.get("objects") or []
    print("returned objects count:", len(objs))
    sample_classes = {}
    for o in objs:
        cls = o.get("class")
        sample_classes[cls] = sample_classes.get(cls, 0) + 1
    print("classes in first page:", sample_classes)
except Exception:
    print(r.text[:2000])
