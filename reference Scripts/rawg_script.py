import os
import requests
from dotenv import load_dotenv

load_dotenv()
RAWG_API_KEY = os.getenv("RAWG_API_KEY")
if not RAWG_API_KEY:
    raise RuntimeError("Missing RAWG_API_KEY in environment")

def search_game_rawg(query, page_size=5):
    endpoint = "https://api.rawg.io/api/games"
    params = {
        "key": RAWG_API_KEY,
        "search": query,
        "page_size": page_size
    }
    resp = requests.get(endpoint, params=params)
    if resp.status_code != 200:
        print("Error querying RAWG search:", resp.status_code, resp.text)
        return []
    data = resp.json()
    return data.get("results", [])

def get_game_details_rawg(game_id):
    endpoint = f"https://api.rawg.io/api/games/{game_id}"
    params = {
        "key": RAWG_API_KEY
    }
    resp = requests.get(endpoint, params=params)
    if resp.status_code != 200:
        print("Error fetching RAWG details:", resp.status_code, resp.text)
        return None
    return resp.json()

if __name__ == "__main__":
    q = input("Enter game title or keyword: ").strip()
    print(f"\nSearching RAWG for '{q}' …\n")
    results = search_game_rawg(q, page_size=5)
    if not results:
        print("No games found in RAWG.")
    else:
        print("Top search results:")
        for i, g in enumerate(results, start=1):
            print(f"{i}. {g.get('name')} (ID: {g.get('id')}) — Released: {g.get('released')}")
        first = results[0]
        print("\nFetching full details for first result …\n")
        details = get_game_details_rawg(first.get("id"))
        if details:
            import json
            print(json.dumps(details, indent=2))
