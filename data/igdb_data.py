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

# Visual fields to remove
VISUAL_KEYS = {"artworks", "cover", "screenshots", "videos"}


# ---------------------------------------
# RAWG — Correct Game Name Resolver
# ---------------------------------------

def _resolve_correct_name(query: str) -> Optional[str]:
    if not RAWG_API_KEY:
        raise ValueError("Missing RAWG_API_KEY in environment!")

    url = "https://api.rawg.io/api/games"
    params = {"key": RAWG_API_KEY, "search": query, "page_size": 1}

    try:
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()
        if data.get("results"):
            return data["results"][0].get("name")
        return None
    except Exception as e:
        raise RuntimeError(f"RAWG error resolving correct name: {e}")


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


def _igdb_post(query: str, token: str) -> List[Dict[str, Any]]:
    headers = {
        "Client-ID": TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {token}",
    }
    resp = requests.post(IGDB_GAMES_URL, headers=headers, data=query, timeout=60)
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
      1. Resolve correct RAWG name
      2. Authenticate with IGDB
      3. Search IGDB using 'fields *'
      4. Optionally remove visual/media fields

    Returns dict:
    {
        "query_name": original user name,
        "resolved_name": name from RAWG,
        "raw": full IGDB records,
        "clean": cleaned records (no visual fields)
    }
    """

    if not query:
        raise ValueError("Query must be a non-empty string.")

    # 1. RAWG resolve name
    resolved = _resolve_correct_name(query)
    if not resolved:
        raise RuntimeError(f"No matching game found on RAWG for {query!r}")

    # 2. IGDB authentication
    token = _get_twitch_token()

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