# tests/list_chunks_for_unified.py
import os
import json
import requests
from vector.index_manager import client, COLLECTION_NAME

WEAVIATE = os.getenv("WEAVIATE_URL", "http://localhost:8080")
UNIFIED = "far-cry-3-2012-2012-aa60e468"

query = {
  "query": """
  {
    Get {
      %s(where: {
        path: ["unified_game_id"],
        operator: Equal,
        valueString: "%s"
      }) {
        chunk_uuid
        chunk_index
        source
        text
        _additional {
          id
        }
      }
    }
  }
  """ % (COLLECTION_NAME, UNIFIED)
}

try:
    resp = requests.post(WEAVIATE.rstrip("/") + "/v1/graphql", json=query, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    print(json.dumps(j, indent=2))
except Exception as e:
    print("GraphQL HTTP query failed:", repr(e))
finally:
    try:
        if hasattr(client, "close"):
            client.close()
    except Exception:
        pass
