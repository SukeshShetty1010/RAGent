# auth/rawg_client.py
import os
import time
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RAWGClient:
    """
    A secure, rate-limited client for the RAWG API.
    Handles authentication via API key and respects rate limits.
    """
    BASE_URL = "https://api.rawg.io/api"
    RATE_LIMIT_SLEEP = 1.1  # RAWG allows ~1 req/sec per key; be safe

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("RAWG_API_KEY")
        if not self.api_key:
            raise RuntimeError("RAWG_API_KEY is missing. Set it in .env or pass explicitly.")
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.RATE_LIMIT_SLEEP:
            time.sleep(self.RATE_LIMIT_SLEEP - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """Internal GET with rate limiting and error handling."""
        self._rate_limit()
        if params is None:
            params = {}
        params["key"] = self.api_key
        url = f"{self.BASE_URL}{endpoint}"

        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                print("[RAWG] Rate limited. Sleeping 5s...")
                time.sleep(5)
                return self._get(endpoint, params)
            if resp.status_code != 200:
                print(f"[RAWG WARN] {url} -> {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json()
        except requests.RequestException as e:
            print(f"[RAWG ERROR] Request failed: {e}")
            return None

    def search_games(self, query: str, page_size: int = 10) -> list:
        """Search games by keyword."""
        data = self._get("/games", {"search": query, "page_size": page_size})
        return data.get("results", []) if data else []

    def get_game_details(self, game_id: int) -> Optional[Dict]:
        """Fetch full game details."""
        return self._get(f"/games/{game_id}")

    def get_game_additions(self, game_id: int) -> Optional[Dict]:
        return self._get(f"/games/{game_id}/additions")

    def get_game_series(self, game_id: int) -> Optional[Dict]:
        return self._get(f"/games/{game_id}/game-series")

    def get_achievements(self, game_id: int) -> Optional[Dict]:
        return self._get(f"/games/{game_id}/achievements")

    def get_stores(self, game_id: int) -> Optional[Dict]:
        return self._get(f"/games/{game_id}/stores")