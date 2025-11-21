# data/gamespot_data.py
"""
GameSpotData — Fetches and structures game data from the GameSpot API
in a hierarchical format for RAGent.

Flow:
1. Use RAWG (via helper.py) to correct the game name.
2. Search GameSpot API using the corrected name.
3. Retrieve detailed data from /games, /releases, /articles, /reviews.
4. Convert into hierarchical JSON structure.
"""

import json
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from auth.gamespot_client import GameSpotClient
from data.rawg_data import RAWGData  # <-- FIX: Moved import to top level

PAGE_LIMIT = 3
VISUAL_KEYS = {"image", "images", "screenshot", "screenshots", "thumbnail", "video_urls", "video_url", "image_tags"}


def remove_visual_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Remove visual-related fields to keep data text-focused."""
    return {k: v for k, v in record.items() if k not in VISUAL_KEYS}


def safe_get(d: Dict[str, Any], *keys):
    """Safely traverse nested dicts."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


class GameSpotData:
    def __init__(self, client: Optional[GameSpotClient] = None):
        self.client = client or GameSpotClient()

    def _get_corrected_name(self, name: str) -> Optional[str]:
        """Use RAWG helper to get the corrected game name."""
        try:
            # from data.rawg_data import RAWGData  <-- FIX: Removed from here
            rawg = RAWGData()
            results = rawg.search_and_rank_games(name, top_k=1)
            if results:
                corrected_name = results[0]["name"]
                print(f"[GameSpotData] Corrected search name → '{corrected_name}'")
                return corrected_name
        except Exception as e:
            print(f"[GameSpotData] RAWG name correction failed: {e}")
        return name

    def _fetch_endpoint(self, name: str, endpoint: str, filter_str: str):
        """Fetch all pages for a given GameSpot endpoint."""
        try:
            data = self.client.fetch_all_pages(endpoint, filter_str, max_pages=PAGE_LIMIT)
            print(f"[GameSpotData] {name}: {len(data)} records fetched")
            return name, [remove_visual_fields(r) for r in data]
        except Exception as e:
            print(f"[WARN] {name} failed: {e}")
            return name, []

    def get_game_data(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch and return structured hierarchical GameSpot data."""
        corrected_name = self._get_corrected_name(title)
        print(f"[GameSpotData] Using corrected name: {corrected_name}")

        # Step 1: Find the game by name
        search_results = self.client.fetch("games", f"name:{corrected_name}", limit=10)
        if not search_results:
            print("[GameSpotData] Game not found on GameSpot.")
            return None

        game = next((g for g in search_results if g.get("name", "").lower() == corrected_name.lower()), search_results[0])
        game_id = game.get("id")
        print(f"[GameSpotData] Found game '{corrected_name}' with id {game_id}")

        # Step 2: Define endpoints to pull
        endpoints = {
            "game": ("games", f"id:{game_id}"),
            "releases": ("releases", f"game:{game_id}"),
            "articles": ("articles", f"game:{game_id}"),
            "reviews": ("reviews", f"game:{game_id}")
        }

        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._fetch_endpoint, name, url, f_str): name
                for name, (url, f_str) in endpoints.items()
            }
            for future in as_completed(futures):
                name, data = future.result()
                results[name] = data

        # Step 3: Convert to hierarchical format
        return self._to_hierarchical(results)

    def _to_hierarchical(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat GameSpot data into hierarchical JSON structure."""
        structured = {
            "Game Information": {},
            "Releases": [],
            "Articles": [],
            "Reviews": []
        }

        if data.get("game"):
            g = data["game"][0]
            structured["Game Information"] = {
                "id": g.get("id"),
                "name": g.get("name"),
                "deck": g.get("deck"),
                "description": g.get("description"),
                "release_date": g.get("release_date"),
                "genres": g.get("genres"),
                "themes": g.get("themes"),
                "developers": g.get("developers"),
                "publishers": g.get("publishers"),
                "platforms": g.get("platforms"),
                "site_detail_url": g.get("site_detail_url"),
                "release_year": g.get("release_date")[:4] if g.get("release_date") else None,
                "aliases": g.get("aliases"),
                "similar_games": g.get("similar_games"),
                "franchise": g.get("franchise"),
                "updated_at": g.get("updated_at")
            }

        for r in data.get("releases", []):
            structured["Releases"].append({
                "id": r.get("id"),
                "name": r.get("name"),
                "platform": safe_get(r, "platform", "name") or r.get("platform"),
                "release_date": r.get("release_date"),
                "region": r.get("region"),
                "publisher": r.get("publisher"),
                "developer": r.get("developer"),
                "site_detail_url": r.get("site_detail_url"),
                "updated_at": r.get("updated_at")
            })

        for a in data.get("articles", []):
            structured["Articles"].append({
                "id": a.get("id"),
                "title": a.get("title"),
                "deck": a.get("deck"),
                "authors": a.get("authors"),
                "publish_date": a.get("publish_date"),
                "update_date": a.get("update_date"),
                "site_detail_url": a.get("site_detail_url"),
                "categories": a.get("categories"),
            })

        for r in data.get("reviews", []):
            structured["Reviews"].append({
                "id": r.get("id"),
                "title": r.get("title"),
                "score": r.get("score"),
                "deck": r.get("deck"),
                "authors": r.get("authors"),
                "publish_date": r.get("publish_date"),
                "update_date": r.get("update_date"),
                "platforms": r.get("platforms"),
                "site_detail_url": r.get("site_detail_url"),
                "release": safe_get(r, "release", "name"),
                "good": r.get("good"),
                "bad": r.get("bad"),
                "bottom_line": r.get("bottom_line"),
            })

        return structured