import json, requests

resp = requests.post(
    "http://localhost:8080/v1/graphql",
    json={"query": "{ Get { GameKnowledge { unified_game_id title _additional { id } } } }"}
)
print(json.dumps(resp.json(), indent=2))
