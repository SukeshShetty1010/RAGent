import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

_TOKEN_CACHE = {"token": None, "expires_at": 0}


def get_igdb_token():
    """Fetch a fresh IGDB access token or use cached one if valid."""
    current_time = time.time()

    # Reuse cached token if not expired
    if _TOKEN_CACHE["token"] and current_time < _TOKEN_CACHE["expires_at"]:
        return _TOKEN_CACHE["token"]

    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }

    resp = requests.post(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    token = data["access_token"]
    expires_in = data["expires_in"]  # usually 60 * 60 * 2 (2 hours)

    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["expires_at"] = current_time + expires_in - 60  # refresh a bit early

    return token