# auth/gamespot_client.py
"""
GameSpotClient — Handles authentication and API communication
with the GameSpot API.
"""

import os
import time
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

class GameSpotClient:
    BASE_URL = "https://www.gamespot.com/api"
    RATE_LIMIT_SLEEP = 1.1  # simple safety delay between calls

    def __init__(self, api_key: Optional[str] = None, user_agent: str = "RAGentGameSpotClient/1.0"):
        self.api_key = api_key or os.getenv("GAMESPOT_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GAMESPOT_API_KEY. Set it in .env or pass explicitly.")
        self.user_agent = user_agent
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Ensure we respect GameSpot's rate limits."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.RATE_LIMIT_SLEEP:
            time.sleep(self.RATE_LIMIT_SLEEP - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Internal GET wrapper for GameSpot API."""
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint.strip('/')}/"
        params.update({"api_key": self.api_key, "format": "json"})

        try:
            resp = requests.get(url, params=params, headers={"User-Agent": self.user_agent}, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[GameSpot ERROR] {endpoint} request failed: {e}")
            return None

    def fetch(self, endpoint: str, filters: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetch a single page of results from a GameSpot endpoint."""
        params = {"filter": filters, "limit": limit, "offset": offset}
        data = self._get(endpoint, params)
        return data.get("results", []) if data else []

    def fetch_all_pages(self, endpoint: str, filters: str, max_pages: int = 3, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch results from up to max_pages pages."""
        all_results, offset = [], 0
        for page in range(max_pages):
            results = self.fetch(endpoint, filters, limit, offset)
            if not results:
                break
            all_results.extend(results)
            offset += len(results)
            total = results[-1].get("number_of_total_results")
            if total and offset >= total:
                break
        return all_results
