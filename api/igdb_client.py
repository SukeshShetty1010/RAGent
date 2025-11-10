# api/igdb_client.py
"""
IGDB Authentication Layer – Token caching + raw request
Used by data/igdb.py and anywhere else in the project
"""
import os
import requests
import time
from typing import Any, List

# Global token cache with safe expiry
TOKEN_CACHE = {"token": None, "expires_at": 0}

def _get_token() -> str:
    """Get valid Twitch OAuth token with 60s early refresh."""
    if time.time() < TOKEN_CACHE["expires_at"]:
        return TOKEN_CACHE["token"]

    client_id = os.getenv("TWITCH_CLIENT_ID")
    client_secret = os.getenv("TWITCH_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Twitch credentials!\n"
            "Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in .env\n"
            "→ https://dev.twitch.com/console/apps"
        )

    resp = requests.post(
        "https://id.twitch.tv/oauth2/token",
        params={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials"
        },
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()

    TOKEN_CACHE["token"] = data["access_token"]
    TOKEN_CACHE["expires_at"] = time.time() + data["expires_in"] - 60
    return data["access_token"]


def igdb_request(endpoint: str, query: str) -> List[dict]:
    """
    Make authenticated POST request to IGDB v4.
    Auto-refreshes token on 401.
    """
    token = _get_token()
    headers = {
        "Client-ID": os.getenv("TWITCH_CLIENT_ID"),
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    url = f"https://api.igdb.com/v4/{endpoint}"
    resp = requests.post(url, data=query, headers=headers, timeout=15)

    # Auto-retry once if token expired
    if resp.status_code == 401:
        print("[IGDB] Token expired, refreshing...")
        token = _get_token()  # Force refresh
        headers["Authorization"] = f"Bearer {token}"
        resp = requests.post(url, data=query, headers=headers, timeout=15)

    resp.raise_for_status()
    return resp.json()