# tests/inspect_objects_rest.py
import os, json, requests
WEAVIATE = os.getenv("WEAVIATE_URL", "http://localhost:8080")
CLASS = "GameKnowledge"

def pretty(resp):
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text[:2000])

print("GET /v1/objects?class=GameKnowledge&limit=20")
r = requests.get(f"{WEAVIATE.rstrip('/')}/v1/objects?class={CLASS}&limit=20", timeout=30)
print("status:", r.status_code)
pretty(r)

print("\nGET /v1/objects?limit=20 (first page)")
r2 = requests.get(f"{WEAVIATE.rstrip('/')}/v1/objects?limit=20", timeout=30)
print("status:", r2.status_code)
pretty(r2)
