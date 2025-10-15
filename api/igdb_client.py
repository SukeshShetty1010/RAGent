import requests
from api.auth import get_igdb_token
from dotenv import load_dotenv
import os

load_dotenv()
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")

BASE_URL = "https://api.igdb.com/v4"


def igdb_request(endpoint: str, query: str):
    """Generic IGDB API call."""
    token = get_igdb_token()
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {token}",
    }

    url = f"{BASE_URL}/{endpoint}"
    resp = requests.post(url, data=query, headers=headers)
    resp.raise_for_status()
    return resp.json()