"""
data/igdb_data.py

Reusable utility to fetch IGDB game metadata for RAG ingestion.
- Resolves the correct game name using RAWG API
- Fetches IGDB data (fields *)
- Optionally strips visual/media fields
- Returns JSON instead of saving to files

Usage:

from data.igdb_data import fetch_igdb_game_data

data = fetch_igdb_game_data("elden ring", strip_visual=True)
"""

import os
import requests
import json
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RAWG API
RAWG_API_KEY = os.getenv("RAWG_API_KEY")

# IGDB API (uses Twitch OAuth)
TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

# Endpoints
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IGDB_GAMES_URL = "https://api.igdb.com/v4/games"
IGDB_PLATFORMS_URL = "https://api.igdb.com/v4/platforms"

# Visual fields to remove
VISUAL_KEYS = {"artworks", "cover", "screenshots", "videos"}


# ---------------------------------------
# Correct Game Name Resolver
# ---------------------------------------

def _resolve_via_rawg(query: str) -> Optional[str]:
    """Optional fail-soft pre-normalizer. RAWG's fuzzy search sometimes
    returns a cleaner title than a raw query would match on IGDB, but
    this must never block IGDB resolution — mirrors the pattern in
    data/gamespot_data.py's resolve_game_name.

    Gated behind rawg_available()'s shared circuit breaker: this runs
    on every game via fetch_igdb_game_data's resolve step, so against a
    black-holed RAWG host it's the single biggest source of dead wait
    across a bulk rebuild.
    """
    from data.rawg_data import rawg_available, record_rawg_success, record_rawg_failure

    if not RAWG_API_KEY or not rawg_available():
        return None

    url = "https://api.rawg.io/api/games"
    params = {"key": RAWG_API_KEY, "search": query, "page_size": 1}

    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
        record_rawg_success()
        if data.get("results"):
            return data["results"][0].get("name")
    except Exception:
        record_rawg_failure()
    return None


def _resolve_via_igdb(query: str, token: str) -> Optional[str]:
    """Resolve a search query to IGDB's own canonical name.

    Uses the same exact-match preference ladder as identity_resolver's
    _igdb_adapter — a plain top-1 pick here previously fed the wrong
    canonical name (e.g. "Elden Ring" -> "Elden Ring Nightreign") into
    the main search below, poisoning every downstream record.

    Also strips a trailing "(YYYY)" qualifier (e.g. TOP_100_GAMES'
    "Resident Evil 4 (2023)") before searching — IGDB's `search`
    returns zero hits on a literal parenthetical suffix.
    """
    from ingest.identity_resolver import select_best_igdb_match, split_trailing_year

    clean_query, _year_hint = split_trailing_year(query)

    try:
        records = _igdb_post(f'fields name; search "{clean_query}"; limit 10;', token)
    except Exception:
        return None

    best = select_best_igdb_match(records, clean_query)
    return best.get("name") if best else None


def _resolve_correct_name(query: str, token: str) -> Optional[str]:
    """
    Resolve a search query to a name IGDB will match well.

    Tries RAWG first (optional, fail-soft); falls back to IGDB's own
    search if RAWG is unavailable, unset, or down. IGDB is never
    hard-blocked by RAWG's outage state.
    """
    resolved = _resolve_via_rawg(query)
    if resolved:
        return resolved
    return _resolve_via_igdb(query, token)


# ---------------------------------------
# IGDB — Authentication + Request
# ---------------------------------------

def _get_twitch_token() -> str:
    if not (TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET):
        raise ValueError("Missing Twitch credentials for IGDB!")

    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": TWITCH_CLIENT_ID,
            "client_secret": TWITCH_CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _igdb_post(query: str, token: str, url: str = IGDB_GAMES_URL) -> List[Dict[str, Any]]:
    headers = {
        "Client-ID": TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {token}",
    }
    resp = requests.post(url, headers=headers, data=query, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------
# Cleaning logic (strip visual fields)
# ---------------------------------------

def _remove_visual_fields(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for item in records:
        cleaned_item = {
            k: v for k, v in item.items() if k not in VISUAL_KEYS
        }
        cleaned.append(cleaned_item)
    return cleaned


# ---------------------------------------
# Public Function — Main IGDB Fetcher
# ---------------------------------------

def fetch_igdb_game_data(
    query: str,
    strip_visual: bool = True,
    limit: int = 200,
) -> Dict[str, Any]:
    """
    Fetch IGDB metadata for a game name.

    Steps:
      1. Authenticate with IGDB
      2. Resolve correct name (RAWG pre-normalize, IGDB fallback)
      3. Search IGDB using 'fields *'
      4. Optionally remove visual/media fields

    Returns dict:
    {
        "query_name": original user name,
        "resolved_name": resolved canonical name,
        "raw": full IGDB records,
        "clean": cleaned records (no visual fields)
    }
    """

    if not query:
        raise ValueError("Query must be a non-empty string.")

    # 1. IGDB authentication
    token = _get_twitch_token()

    # 2. Resolve name
    resolved = _resolve_correct_name(query, token)
    if not resolved:
        raise RuntimeError(f"No matching game found on IGDB for {query!r}")

    # 3. IGDB fetch
    igdb_query = f"""
        fields *;
        search "{resolved}";
        limit {limit};
    """

    raw_records = _igdb_post(igdb_query, token)

    # 4. Strip visuals if enabled
    clean_records = (
        _remove_visual_fields(raw_records) if strip_visual else raw_records
    )

    return {
        "query_name": query,
        "resolved_name": resolved,
        #"raw": raw_records,
        "clean": clean_records,
    }


# ---------------------------------------
# Execution & File Saving
# ---------------------------------------

if __name__ == "__main__":
    user_input = input("Game name: ").strip()
    
    try:
        print(f"Fetching data for '{user_input}'...")
        result = fetch_igdb_game_data(user_input, strip_visual=True)
        
        # Determine filename based on the resolved name (more accurate than input)
        resolved_name = result.get("resolved_name", user_input)
        safe_name = re.sub(r'[^\w\-]', '_', resolved_name).lower()
        filename = f"igdb_data_{safe_name}.json"
        
        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        print(f"\n[Success] Data saved to '{filename}'")
        print(f"Resolved Name: {resolved_name}")
        print(f"Records found: {len(result['clean'])}")
        
    except Exception as e:
        print(f"\n[Error] {e}")