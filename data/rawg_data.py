"""
data/rawg_data.py

Utility function to fetch game data from RAWG API and return either the full JSON
or a 'stripped' version with visual/media fields removed (for ingestion into RAG).

Usage:
from data.rawg_data import fetch_rawg_game_data
game = fetch_rawg_game_data("elden ring")
"""

import os
import time
import requests
import json
import re
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from copy import deepcopy

load_dotenv()

RAWG_BASE = "https://api.rawg.io/api"


class RawgError(Exception):
    pass


def _default_api_key() -> Optional[str]:
    return os.getenv("RAWG_API_KEY")


def _prune_visual_fields(obj: Any) -> Any:
    """
    Recursively remove fields that are likely visual/media: keys containing
    substrings like 'image', 'screenshot', 'clip', 'movie', 'logo', 'thumbnail'.
    This is a conservative filter intended for ingestion pipelines where images
    are unnecessary.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = k.lower()
            # If key looks visual, skip it entirely
            if any(
                sub in lk
                for sub in (
                    "image",
                    "screenshot",
                    "clip",
                    "movie",
                    "logo",
                    "thumbnail",
                    "background",
                    "icon",
                    "cover",
                )
            ):
                continue
            out[k] = _prune_visual_fields(v)
        return out

    elif isinstance(obj, list):
        return [_prune_visual_fields(x) for x in obj]

    return obj


def _safe_get(url: str, params: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise RawgError(f"Network error calling RAWG: {e}")

    if resp.status_code == 404:
        raise RawgError("RAWG resource not found (404)")

    try:
        data = resp.json()
    except ValueError:
        raise RawgError("RAWG returned non-JSON response")

    if resp.status_code >= 400:
        msg = data.get("detail") if isinstance(data, dict) else None
        raise RawgError(f"RAWG API error {resp.status_code}: {msg or resp.text}")

    return data


def fetch_rawg_game_data(
    query: str,
    api_key: Optional[str] = None,
    strip_visual: bool = True,
    prefer_exact_match: bool = True,
    rate_limit_sleep: float = 0.35,
) -> Dict[str, Any]:
    """
    Search RAWG for `query` and return the game's full details.

    Args:
        query: search string for the game
        api_key: optional override RAWG API key
        strip_visual: remove visual/media fields
        prefer_exact_match: use exact-case-insensitive match if available
        rate_limit_sleep: small delay to avoid API hammering

    Returns:
        Dict with RAWG /games/{id} details (possibly stripped)
    """

    if not query:
        raise ValueError("query must be a non-empty string")

    key = api_key or _default_api_key()
    if not key:
        raise RawgError(
            "RAWG API key not provided. Set RAWG_API_KEY in environment or pass api_key"
        )

    # 1) Search
    search_url = f"{RAWG_BASE}/games"
    params = {"search": query, "page_size": 5, "key": key}

    search_res = _safe_get(search_url, params)
    results = search_res.get("results") or []

    if not results:
        raise RawgError(f"No RAWG results for query: {query!r}")

    chosen = results[0]

    if prefer_exact_match:
        ql = query.strip().lower()
        for r in results:
            name = (r.get("name") or "").strip().lower()
            if name == ql:
                chosen = r
                break

    game_id = chosen.get("id")
    if not game_id:
        raise RawgError("Selected RAWG search result had no 'id' field")

    # RAWG rate limiting protection
    time.sleep(rate_limit_sleep)

    # 2) Fetch full details
    details_url = f"{RAWG_BASE}/games/{game_id}"
    details = _safe_get(details_url, {"key": key})

    if strip_visual:
        return _prune_visual_fields(deepcopy(details))

    return details


if __name__ == "__main__":
    q = input("Game query: ").strip()
    try:
        # Fetch data
        g = fetch_rawg_game_data(q, strip_visual=True)
        
        # Print to console
        import pprint
        pprint.pprint(g)

        # --- Save to JSON File ---
        # Create a safe filename (replace spaces/special chars with underscores)
        safe_name = re.sub(r'[^\w\-]', '_', q).lower()
        filename = f"rawg_data_{safe_name}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(g, f, indent=4, ensure_ascii=False)
        
        print(f"\n[Success] Data saved to '{filename}'")

    except Exception as e:
        print("ERROR:", e)