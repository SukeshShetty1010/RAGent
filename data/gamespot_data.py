# data/gamespot_data.py
"""
gamespot_data.py
----------------
Refactored from the original script to provide a reusable function:
    fetch_gamespot_data(query, *, save=True, output_dir='.', ...)
which returns a dict with the merged (optionally visual-stripped) GameSpot data.

Environment variables used:
  - GAMESPOT_API_KEY  (required unless passed to function)
  - RAWG_API_KEY      (optional; used for name normalization if use_rawg=True)

Example:
    from data.gamespot_data import fetch_gamespot_data
    data = fetch_gamespot_data("Dark Souls", save=True, output_dir=".", strip_visual=True)
"""
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
from dotenv import load_dotenv

# re-load env when module is imported (safe)
load_dotenv()

# ---- Defaults & constants ----
BASE_URL = "https://www.gamespot.com/api"
USER_AGENT = "RAG-ent-GamespotFetcher/1.0 (contact: you@example.com)"

VISUAL_KEY_SUBSTRINGS = [
    "image",
    "images",
    "video",
    "videos",
    "screenshot",
    "thumbnail",
    "thumbnails",
    "poster",
    "banner",
    "icon",
    "avatar",
    "logo",
]


# ---- RAWG helper (unchanged logic, but made robust) ----
def get_correct_game_name_from_rawg(query: str, rawg_api_key: str, page_size: int = 1) -> Optional[str]:
    """
    Query RAWG API for the closest match and return its official name.
    Returns None if RAWG can't provide a match or if there is an error.
    """
    if not rawg_api_key:
        return None

    url = "https://api.rawg.io/api/games"
    params = {"key": rawg_api_key, "search": query, "page_size": page_size}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if results:
            return results[0].get("name")
        return None
    except requests.RequestException:
        # Don't raise here — we gracefully fall back to the raw query
        return None


def resolve_game_name(raw_query: str, rawg_api_key: Optional[str]) -> str:
    """
    Attempt to normalize the game name using RAWG. If RAWG isn't available or returns
    nothing useful, fall back to the original query.
    """
    if rawg_api_key:
        corrected = get_correct_game_name_from_rawg(raw_query, rawg_api_key)
        if corrected:
            # keep the user notified when used interactively
            try:
                print(f"✔ Using RAWG-corrected name: '{corrected}' (from '{raw_query}')")
            except Exception:
                pass
            return corrected

    try:
        print(f"[INFO] Using raw query as resolved name: '{raw_query}'")
    except Exception:
        pass
    return raw_query


# ---- Visual-stripping helpers ----
def is_visual_key(key: str) -> bool:
    k = key.lower()
    return any(sub in k for sub in VISUAL_KEY_SUBSTRINGS)


def strip_visual_fields(obj: Any) -> Any:
    """
    Recursively remove keys from dicts whose key name looks like visual content.
    Works for nested dicts and lists.
    """
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            if is_visual_key(k):
                continue
            cleaned[k] = strip_visual_fields(v)
        return cleaned
    if isinstance(obj, list):
        return [strip_visual_fields(x) for x in obj]
    return obj


# ---- URL & pagination helpers ----
def add_api_params_to_url(url: str, api_key: str, extra_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Append (or merge) api_key, format=json and extra_params into an existing URL.
    """
    if extra_params is None:
        extra_params = {}
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    # Flatten single-valued lists
    merged = {k: v[-1] if isinstance(v, list) else v for k, v in query.items()}
    merged.setdefault("format", "json")
    merged.setdefault("api_key", api_key)
    merged.update(extra_params)
    new_query = urlencode(merged, doseq=False)
    new_parsed = parsed._replace(query=new_query)
    return urlunparse(new_parsed)


def fetch_all_pages_from_url(
    raw_url: str,
    api_key: str,
    params: Optional[Dict[str, Any]] = None,
    delay_sec: float = 0.5,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generic pagination helper for GameSpot-style endpoints.
    Returns list of result objects (may be partial on repeated failures).
    """
    if params is None:
        params = {}
    all_results: List[Dict[str, Any]] = []
    offset = int(params.get("offset", 0))
    limit = int(params.get("limit", 100))

    while True:
        page_params = dict(params)
        page_params["offset"] = offset
        page_params["limit"] = limit

        url = add_api_params_to_url(raw_url, api_key, page_params)

        attempt = 0
        backoff = 1.0
        last_error = None
        data = None

        while attempt < max_retries:
            try:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)

                if resp.status_code == 503:
                    last_error = f"503 Service Unavailable for {url}"
                    attempt += 1
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                resp.raise_for_status()
                try:
                    data = resp.json()
                except ValueError as e:
                    last_error = f"JSON decode error for {url}: {e}"
                    attempt += 1
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                break

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = f"{type(e).__name__} while fetching {url}: {e}"
                attempt += 1
                time.sleep(backoff)
                backoff *= 2
            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (500, 502, 503, 504):
                    last_error = f"HTTP {status} while fetching {url}: {e}"
                    attempt += 1
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    # Non-retryable
                    return all_results
        else:
            # Retries exhausted for this page
            return all_results

        page_results = data.get("results", []) or []
        all_results.extend(page_results)

        number_of_page_results = data.get("number_of_page_results", len(page_results))
        number_of_total_results = data.get("number_of_total_results", len(page_results))

        if number_of_page_results == 0:
            break

        offset += number_of_page_results
        if offset >= number_of_total_results:
            break

        time.sleep(delay_sec)

    return all_results


# ---- GameSpot-specific fetchers ----
def fetch_games_by_name(api_key: str, name: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch all games from /games/ matching the provided name filter.
    """
    url = f"{BASE_URL}/games/"
    params = {
        "format": "json",
        "api_key": api_key,
        "filter": f"name:{name}",
        "limit": limit,
        "offset": 0,
    }
    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    games = data.get("results", []) or []

    total = data.get("number_of_total_results", len(games))
    page_count = data.get("number_of_page_results", len(games))
    offset = data.get("offset", 0)
    limit = data.get("limit", len(games))

    all_games = list(games)
    while offset + page_count < total:
        offset += page_count
        params["offset"] = offset
        params["limit"] = limit
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        games = data.get("results", []) or []
        page_count = data.get("number_of_page_results", len(games))
        all_games.extend(games)
        if page_count == 0:
            break
        time.sleep(0.5)

    return all_games


def fetch_related_textual_data_for_game(game_obj: Dict[str, Any], api_key: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    For a single /games/ object, fetch textual related endpoints (reviews, articles, releases).
    Skips images/videos endpoints.
    """
    related: Dict[str, List[Dict[str, Any]]] = {}
    reviews_url = game_obj.get("reviews_api_url")
    articles_url = game_obj.get("articles_api_url")
    releases_url = game_obj.get("releases_api_url")

    related["reviews"] = fetch_all_pages_from_url(reviews_url, api_key) if reviews_url else []
    related["articles"] = fetch_all_pages_from_url(articles_url, api_key) if articles_url else []
    related["releases"] = fetch_all_pages_from_url(releases_url, api_key) if releases_url else []

    return related


# ---- Top-level function intended for project use ----
def fetch_gamespot_data(
    query: str,
    *,
    gamespot_api_key: Optional[str] = None,
    rawg_api_key: Optional[str] = None,
    use_rawg: bool = True,
    strip_visual: bool = True,
    save: bool = False,
    output_dir: str = ".",
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch GameSpot data for a query and return a merged dict.

    Args:
      query: the user/query string (e.g., "Elden Ring")
      gamespot_api_key: if None, read from GAMESPOT_API_KEY env var
      rawg_api_key: if None, read from RAWG_API_KEY env var (only used if use_rawg=True)
      use_rawg: whether to attempt name normalization via RAWG
      strip_visual: whether to remove visual-related fields from the final structure
      save: whether to save the resulting JSON to disk
      output_dir: where to save the JSON
      filename: explicit filename (if provided, used as-is), otherwise generated from resolved name

    Returns:
      A dict with the merged data (same structure as your original script's `merged` variable).
    """
    # Resolve keys from env if not provided
    gamespot_api_key = gamespot_api_key or os.getenv("GAMESPOT_API_KEY")
    if not gamespot_api_key:
        raise RuntimeError("GAMESPOT_API_KEY is required (either pass as argument or set in .env)")

    if use_rawg:
        rawg_api_key = rawg_api_key or os.getenv("RAWG_API_KEY")

    # Resolve game name
    resolved_name = resolve_game_name(query, rawg_api_key if use_rawg else None)

    # Fetch games
    games = fetch_games_by_name(gamespot_api_key, resolved_name)
    merged: Dict[str, Any] = {
        "query": {"requested": query, "resolved": resolved_name},
        "games_count": len(games),
        "games": [],
    }

    for game in games:
        related_textual = fetch_related_textual_data_for_game(game, gamespot_api_key)
        merged["games"].append({"game": game, "related": related_textual})

    # Optionally strip visual fields
    result = strip_visual_fields(merged) if strip_visual else merged

    # Save if requested
    if save:
        safe_base = (resolved_name or query).lower().replace(" ", "_")
        out_name = filename or f"{safe_base}_gamespot_full_textual.json"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        try:
            print(f"Saved GameSpot data to: {out_path}")
        except Exception:
            pass

    return result


# ---- Convenience CLI (keeps backward compatibility) ----
if __name__ == "__main__":
    load_dotenv()

    gamespot_api_key = os.getenv("GAMESPOT_API_KEY")
    rawg_api_key = os.getenv("RAWG_API_KEY")

    if not gamespot_api_key:
        print("❌ ERROR: GAMESPOT_API_KEY is not set in your environment (.env).")
        exit(1)

    # Ask interactively for the game name
    raw_query = input("Enter the game name to fetch data for: ").strip()

    if not raw_query:
        print("❌ You must enter a game name.")
        exit(1)

    print(f"Fetching GameSpot data for '{raw_query}' (textual only)...")
    fetch_gamespot_data(
        raw_query,
        gamespot_api_key=gamespot_api_key,
        rawg_api_key=rawg_api_key,
        save=True
    )