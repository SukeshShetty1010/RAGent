"""
data/steam_data.py

Reusable utility to fetch Steam store page data for RAG ingestion. No
API key required — public storefront endpoints.

Usage:
    from data.steam_data import fetch_steam_game_data
    data = fetch_steam_game_data("Elden Ring")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
DETAILS_URL = "https://store.steampowered.com/api/appdetails"
TIMEOUT = 15


def find_steam_appid(name: str) -> Optional[int]:
    """Resolve a game name to a Steam store appid via storesearch."""
    if not name or not name.strip():
        return None
    try:
        resp = requests.get(
            SEARCH_URL, params={"term": name, "cc": "us", "l": "en"}, timeout=TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    items = data.get("items") or []
    if not items:
        return None
    return items[0].get("id")


def fetch_steam_appdetails(appid: int) -> Optional[Dict[str, Any]]:
    """Fetch the full appdetails payload for a known Steam appid."""
    try:
        resp = requests.get(
            DETAILS_URL, params={"appids": appid, "l": "en"}, timeout=TIMEOUT
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None

    entry = (payload or {}).get(str(appid)) or {}
    if not entry.get("success"):
        return None
    return entry.get("data")


def fetch_steam_game_data(name: str) -> Optional[Dict[str, Any]]:
    """
    Resolve `name` to a Steam appid and fetch its store page data.

    Returns the raw appdetails `data` dict (name, detailed_description,
    about_the_game, pc_requirements, platforms, release_date,
    metacritic, ...) with `_appid` injected, or None if unresolvable.
    Used both for editorial content (ingest/steam_editorial_normalize.py)
    and PlatformSpec hardware requirements (ingest/platform_sources.py).
    """
    appid = find_steam_appid(name)
    if not appid:
        return None

    data = fetch_steam_appdetails(appid)
    if not data:
        return None

    data["_appid"] = appid
    return data
