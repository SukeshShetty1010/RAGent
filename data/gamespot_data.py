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
import logging
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from auth.gamespot_client import GameSpotClient
from data.rawg_data import RAWGData  # <-- keep top-level import

PAGE_LIMIT = 3
VISUAL_KEYS = {"image", "images", "screenshot", "screenshots", "thumbnail", "video_urls", "video_url", "image_tags"}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def is_record_for_game(record: Dict[str, Any], game_id: Optional[int], game_title: Optional[str], game_slug: Optional[str]) -> bool:
    """
    Public helper used by tests. Implements strict->fallback matching.
    - Strict: if the record contains an explicit game id (common field names),
      and it equals the game_id provided, return True.
    - Fallback: if strict fails, if the record's name/ title or site_detail_url
      contains the game's title or slug (substring match), return True.
    """
    # Strict game-id based matching: try common fields
    if record is None:
        return False

    # Try to find a game association inside the record under several possible keys
    try_fields = ["game", "game_id", "gameId", "gameIds", "games", "associated_game"]
    for f in try_fields:
        val = record.get(f) if isinstance(record, dict) else None
        if val is None:
            continue
        # normalize list or single val
        if isinstance(val, list) and val:
            # if list of dicts that contain 'id', check those
            if isinstance(val[0], dict) and "id" in val[0]:
                try:
                    ids = [int(x.get("id")) for x in val if x.get("id") is not None]
                    if game_id and int(game_id) in ids:
                        return True
                except Exception:
                    pass
            else:
                # list of ids?
                try:
                    ids = [int(x) for x in val if x is not None]
                    if game_id and int(game_id) in ids:
                        return True
                except Exception:
                    pass
        else:
            # single value
            try:
                if game_id is not None and int(val) == int(game_id):
                    return True
            except Exception:
                pass

    # If strict matching failed, try fallback matching on strings
    name_candidates = []
    for k in ("name", "title", "deck", "site_detail_url"):
        v = record.get(k)
        if v:
            name_candidates.append(_normalize_text(str(v)))

    # normalized title/slug
    n_title = _normalize_text(game_title)
    n_slug = _normalize_text(game_slug)

    # If any name candidate contains title or slug, consider it a match
    for cand in name_candidates:
        if n_title and n_title in cand:
            return True
        if n_slug and n_slug in cand:
            return True

    # As a last check, if the record has a URLs that contain the slug
    url = _normalize_text(record.get("site_detail_url") or record.get("url") or "")
    if n_slug and n_slug in url:
        return True

    return False


class GameSpotData:
    def __init__(self, client: Optional[GameSpotClient] = None):
        self.client = client or GameSpotClient()

    def _get_corrected_name(self, name: str) -> Optional[str]:
        """Use RAWG helper to get the corrected game name."""
        try:
            rawg = RAWGData()
            results = rawg.search_and_rank_games(name, top_k=1)
            if results:
                corrected_name = results[0]["name"]
                logger.info("[GameSpotData] Corrected search name → '%s'", corrected_name)
                return corrected_name
        except Exception as e:
            logger.warning("[GameSpotData] RAWG name correction failed: %s", e)
        return name

    def _fetch_endpoint(self, name: str, endpoint: str, filter_str: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Fetch all pages for a given GameSpot endpoint."""
        try:
            data = self.client.fetch_all_pages(endpoint, filter_str, max_pages=PAGE_LIMIT)
            logger.info("[GameSpotData] %s: %d records fetched", name, len(data))
            return name, [remove_visual_fields(r) for r in data]
        except Exception as e:
            logger.warning("[WARN] %s failed: %s", name, e)
            return name, []

    def get_game_data(self, title: str, low_result_fallback_threshold: int = 2) -> Optional[Dict[str, Any]]:
        """Fetch and return structured hierarchical GameSpot data."""
        corrected_name = self._get_corrected_name(title)
        logger.info("[GameSpotData] Using corrected name: %s", corrected_name)

        # Step 1: Find the game by name
        search_results = self.client.fetch("games", f"name:{corrected_name}", limit=10)
        if not search_results:
            logger.info("[GameSpotData] Game not found on GameSpot.")
            return None

        game = next((g for g in search_results if g.get("name", "").lower() == corrected_name.lower()), search_results[0])
        game_id = game.get("id")
        game_slug = game.get("site_detail_url", "").rstrip("/").split("/")[-1] if game.get("site_detail_url") else corrected_name.lower().replace(" ", "-")
        logger.info("[GameSpotData] Found game '%s' with id %s", corrected_name, game_id)

        # Step 2: Define endpoints to pull (attempt to use game:<id> filters first)
        endpoints = {
            "game": ("games", f"id:{game_id}"),
            "releases": ("releases", f"game:{game_id}"),
            "articles": ("articles", f"game:{game_id}"),
            "reviews": ("reviews", f"game:{game_id}")
        }

        results: Dict[str, List[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._fetch_endpoint, name, url, f_str): name
                for name, (url, f_str) in endpoints.items()
            }
            for future in as_completed(futures):
                name, data = future.result()
                results[name] = data

        # Step 3: Convert to hierarchical format
        hierarchical = self._to_hierarchical(results, game_id=game_id, game_title=corrected_name, game_slug=game_slug, low_result_fallback_threshold=low_result_fallback_threshold)
        return hierarchical

    def _to_hierarchical(self, data: Dict[str, Any], game_id: Optional[int], game_title: Optional[str], game_slug: Optional[str], low_result_fallback_threshold: int = 2) -> Dict[str, Any]:
        """Convert flat GameSpot data into hierarchical JSON structure and apply filtering."""

        structured = {
            "Game Information": {},
            "Releases": [],
            "Articles": [],
            "Reviews": []
        }

        # Populate Game Information
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

        # Helper: initial raw lists (before filtering)
        raw_releases = data.get("releases", []) or []
        raw_articles = data.get("articles", []) or []
        raw_reviews = data.get("reviews", []) or []

        # Strict filtering pass (game-id based and exact matches)
        strict_releases = [r for r in raw_releases if is_record_for_game(r, game_id, None, None)]
        strict_articles = [a for a in raw_articles if is_record_for_game(a, game_id, None, None)]
        strict_reviews = [rv for rv in raw_reviews if is_record_for_game(rv, game_id, None, None)]

        # If strict yields too few results, fallback to relaxed title/slug/url matching
        def maybe_fallback(strict_list, raw_list, kind):
            if len(strict_list) < low_result_fallback_threshold:
                # apply fallback that considers title and slug
                fallback_list = [r for r in raw_list if is_record_for_game(r, None, game_title, game_slug)]
                logger.info("[GameSpotData] %s strict kept %d; fallback kept %d", kind, len(strict_list), len(fallback_list))
                # If fallback yields more, use that; else keep strict
                if len(fallback_list) > len(strict_list):
                    return fallback_list
            return strict_list

        kept_releases = maybe_fallback(strict_releases, raw_releases, "releases")
        kept_articles = maybe_fallback(strict_articles, raw_articles, "articles")
        kept_reviews = maybe_fallback(strict_reviews, raw_reviews, "reviews")

        # Build structured lists (trim fields to text-focused items)
        for r in kept_releases:
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

        for a in kept_articles:
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

        for rv in kept_reviews:
            structured["Reviews"].append({
                "id": rv.get("id"),
                "title": rv.get("title"),
                "score": rv.get("score"),
                "deck": rv.get("deck"),
                "authors": rv.get("authors"),
                "publish_date": rv.get("publish_date"),
                "update_date": rv.get("update_date"),
                "platforms": rv.get("platforms"),
                "site_detail_url": rv.get("site_detail_url"),
                "release": safe_get(rv, "release", "name"),
                "good": rv.get("good"),
                "bad": rv.get("bad"),
                "bottom_line": rv.get("bottom_line"),
            })

        # Attach counts: raw & filtered (per your preference)
        structured["Releases_count_raw"] = len(raw_releases)
        structured["Releases_count"] = len(kept_releases)

        structured["Articles_count_raw"] = len(raw_articles)
        structured["Articles_count"] = len(kept_articles)

        structured["Reviews_count_raw"] = len(raw_reviews)
        structured["Reviews_count"] = len(kept_reviews)

        logger.info("[GameSpotData] Releases: fetched %d kept %d for game_id=%s", len(raw_releases), len(kept_releases), game_id)
        logger.info("[GameSpotData] Articles: fetched %d kept %d for game_id=%s", len(raw_articles), len(kept_articles), game_id)
        logger.info("[GameSpotData] Reviews: fetched %d kept %d for game_id=%s", len(raw_reviews), len(kept_reviews), game_id)

        return structured
