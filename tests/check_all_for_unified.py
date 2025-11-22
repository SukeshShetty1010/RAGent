import os, json, requests
from vector.index_manager import COLLECTION_NAME

WEAVIATE = "http://localhost:8080"
UNIFIED = "far-cry-3-2012-2012-aa60e468"

query = {
  "query": f"""
  {{
    Get {{
      {COLLECTION_NAME}(where: {{
        path: ["unified_game_id"],
        operator: Equal,
        valueString: "{UNIFIED}"
      }}) {{
        title
        source
        text
        content_length
        game_id
        slug
        unified_game_id
        _additional {{
          id
        }}
      }}
    }}
  }}
  """
}

resp = requests.post(WEAVIATE + "/v1/graphql", json=query)
print(json.dumps(resp.json(), indent=2))
