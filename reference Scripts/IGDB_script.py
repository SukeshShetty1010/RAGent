import os
import requests
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("Missing Twitch credentials")

def get_token():
    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials"
    }
    resp = requests.post(url, params=params)
    resp.raise_for_status()
    return resp.json()["access_token"]

def fetch_game_data(query, limit=3):
    token = get_token()
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}"
    }
    endpoint = "https://api.igdb.com/v4/games"

    # 1) Try exact search
    body1 = f'search "{query}"; fields *; limit {limit};'
    resp1 = requests.post(endpoint, headers=headers, data=body1)
    if resp1.status_code == 200:
        results1 = resp1.json()
        if results1:
            return results1

    # 2) Try wildcard / partial match
    normalized = query.replace("’", "'").strip()
    body2 = f'fields *; where name ~ *"{normalized}"*; limit {limit};'
    resp2 = requests.post(endpoint, headers=headers, data=body2)
    if resp2.status_code == 200:
        results2 = resp2.json()
        if results2:
            return results2

    # 3) Fallback to last word keyword
    keywords = normalized.split()
    if keywords:
        key = keywords[-1]
        body3 = f'fields *; where name ~ *"{key}"*; limit {limit};'
        resp3 = requests.post(endpoint, headers=headers, data=body3)
        if resp3.status_code == 200:
            return resp3.json()

    # No result
    return []

if __name__ == "__main__":
    q = input("Enter game title or keyword: ").strip()
    print(f"\nFetching full IGDB data for '{q}' …\n")
    results = fetch_game_data(q, limit=3)
    if not results:
        print("No results found.")
    else:
        import json
        print(json.dumps(results, indent=2))
