# auth/igdb_client.py
"""
IGDBClient — handles Twitch OAuth and IGDB API requests.
Used by data/igdb_data.py for authenticated IGDB data access.
"""

import os
import time
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()


class IGDBClient:
    BASE_URL = "https://api.igdb.com/v4"
    TOKEN_URL = "https://id.twitch.tv/oauth2/token"
    RATE_LIMIT_SLEEP = 1.1  # keep it gentle for IGDB rate limits

    def __init__(self,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None):
        self.client_id = client_id or os.getenv("TWITCH_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("TWITCH_CLIENT_SECRET")
        if not self.client_id or not self.client_secret:
            raise RuntimeError("Missing Twitch credentials. Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in .env")

        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._last_request_time: float = 0.0

    # ---------- Authentication ----------

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.RATE_LIMIT_SLEEP:
            time.sleep(self.RATE_LIMIT_SLEEP - elapsed)
        self._last_request_time = time.time()

    def _fetch_new_token(self) -> str:
        params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }
        resp = requests.post(self.TOKEN_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + int(data.get("expires_in", 3600)) - 30
        return self._token

    def get_token(self) -> str:
        if not self._token or time.time() > self._token_expiry:
            return self._fetch_new_token()
        return self._token

    # ---------- API core ----------

    def post(self, endpoint: str, body: str) -> List[Dict[str, Any]]:
        """
        Generic IGDB POST call. Handles token refresh and rate limiting.
        """
        self._rate_limit()
        token = self.get_token()
        headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {token}",
        }
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            resp = requests.post(url, headers=headers, data=body, timeout=15)
            if resp.status_code == 401:
                # token expired mid-flight — refresh once
                self._fetch_new_token()
                headers["Authorization"] = f"Bearer {self._token}"
                resp = requests.post(url, headers=headers, data=body, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[IGDB ERROR] {e}")
            return []
