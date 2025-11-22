# tests/http_batch_test.py
import json, os, requests
from pathlib import Path
WEAVIATE = os.getenv("WEAVIATE_URL", "http://localhost:8080")
CLASS = "GameKnowledge"

base = Path(".")
canon = json.load(open(base / "merged_canonical.json", "r", encoding="utf-8"))
docs = json.load(open(base / "all_docs.json", "r", encoding="utf-8"))

# prepare objects similar to upsert.prepare
objects = []
# canonical
obj_id = canon.get("unified_game_id") or canon.get("slug")
objects.append({"class": CLASS, "id": obj_id, "properties": {k:v for k,v in canon.items()}})
# add each doc as an object
for d in docs:
    props = {}
    # combine metadata + top-level text into properties
    meta = d.get("metadata", {})
    props.update({k:v for k,v in meta.items()})
    props["text"] = d.get("text")
    # deterministic uuid per chunk (match upsert's logic if needed)
    chunk_id = f"{props.get('unified_game_id') or props.get('parent_unified_id') or 'unknown'}__chunk__{props.get('chunk_uuid') or props.get('chunk_index')}"
    obj_uuid = chunk_id
    objects.append({"class": CLASS, "id": obj_uuid, "properties": props})

payload = {"objects": objects}
print("Posting payload with %d objects to %s" % (len(objects), WEAVIATE.rstrip("/") + "/v1/batch/objects"))
resp = requests.post(WEAVIATE.rstrip("/") + "/v1/batch/objects", json=payload, timeout=120)
print("status:", resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2)[:4000])
except Exception:
    print(resp.text[:4000])
