# data/helper.py
"""
Helper utilities used across the ingestion pipeline.

Contains:
- datetime helpers (RFC3339 Z)
- simple hashing (8 hex)
- safe int conversion
- list normalization (flattening/dedup)
- platform canonicalization
- small review/article helpers (extract game id)
"""

import json
import hashlib
from typing import Any, List, Optional, Dict
from datetime import datetime, timezone
import re

# -----------------------
# Time / Hash / Conversions
# -----------------------
def now_iso_z() -> str:
    """Return current time in RFC3339 Z format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_hash8(s: str) -> str:
    """Return first 8 hex chars of sha1(s). Deterministic short id."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def to_int_or_none(v: Any) -> Optional[int]:
    """Safely cast value to int, or return None."""
    if v is None:
        return None
    try:
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


# -----------------------
# List normalization
# -----------------------
def normalize_simple_list(value: Any) -> List[str]:
    """
    Normalize various array-like inputs into a List[str]:
      - None -> []
      - list of strings or dicts -> extract 'name'/'title' or stringify
      - JSON-encoded list string -> parse, then normalize
      - comma-separated string -> split
    Uses case-insensitive dedupe while preserving first-seen casing.
    """
    if value is None:
        return []

    parts = []
    # If it's a JSON list string, try parse first
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith('"') and s.endswith('"')):
            try:
                parsed = json.loads(s)
                return normalize_simple_list(parsed)
            except Exception:
                pass
        # If it contains commas, split
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
        else:
            parts = [s] if s else []
    elif isinstance(value, (list, tuple, set)):
        for it in value:
            if it is None:
                continue
            if isinstance(it, dict):
                name = it.get("name") or it.get("title") or it.get("slug")
                if name:
                    parts.append(str(name).strip())
                else:
                    # fallback: stringify dict compactly
                    try:
                        parts.append(json.dumps(it, ensure_ascii=False))
                    except Exception:
                        parts.append(str(it))
            else:
                parts.append(str(it).strip())
    else:
        parts = [str(value).strip()]

    # Deduplicate case-insensitively preserving first seen
    seen = set()
    out = []
    for p in parts:
        k = p.lower()
        if k not in seen and p != "":
            seen.add(k)
            out.append(p)
    return out


# -----------------------
# Platform canonicalization
# -----------------------
# Small canonical map; expand as needed
_PLATFORM_MAP = {
    "pc (microsoft windows)": "PC",
    "pc (windows)": "PC",
    "pc": "PC",
    "playstation 4": "PS4",
    "ps4": "PS4",
    "playstation 5": "PS5",
    "ps5": "PS5",
    "xbox one": "Xbox One",
    "xbox series x": "Xbox Series X",
    "nintendo switch": "Switch",
    "switch": "Switch",
}


def canonicalize_platforms(platforms: List[str]) -> List[str]:
    """Canonicalize platforms to a small set of common labels and dedupe."""
    out = []
    seen = set()
    for p in platforms:
        if not p:
            continue
        pk = p.strip().lower()
        val = _PLATFORM_MAP.get(pk)
        if not val:
            # Strip parenthetical qualifiers: "PC (Microsoft Windows)" -> "PC"
            if "(" in p:
                val = p.split("(")[0].strip()
            else:
                val = p.strip()
        key = val.lower()
        if key not in seen:
            seen.add(key)
            out.append(val)
    return out


# -----------------------
# Review/article helper
# -----------------------
def extract_game_id_from_item(item: Any) -> Optional[int]:
    """
    Try several heuristics to extract related game id from a review/article item:
      - item.get('game') if dict with 'id'
      - item.get('game_id') or item.get('gameId') or item.get('id_game')
      - nested structures (release -> game -> id)
    Returns int or None.
    """
    if item is None:
        return None
    # If item is dict look for common keys
    if isinstance(item, dict):
        # game: {id: 123}
        g = item.get("game")
        if isinstance(g, dict) and g.get("id") is not None:
            return to_int_or_none(g.get("id"))
        # direct game id fields
        for key in ("game_id", "gameId", "gameId", "gameid", "id_game", "gameID"):
            if key in item and item.get(key) is not None:
                return to_int_or_none(item.get(key))
        # release -> game -> id
        rel = item.get("release") or item.get("releases")
        if isinstance(rel, dict):
            gg = rel.get("game")
            if isinstance(gg, dict) and gg.get("id") is not None:
                return to_int_or_none(gg.get("id"))
        # try data.attributes.game.id (common in some nested responses)
        try:
            attrs = item.get("attributes")
            if isinstance(attrs, dict):
                gg = attrs.get("game")
                if isinstance(gg, dict) and gg.get("id") is not None:
                    return to_int_or_none(gg.get("id"))
        except Exception:
            pass
    # fallback None
    return None


def site_url_contains_slug(url: Optional[str], slug: Optional[str]) -> bool:
    """Return True if slug appears in url (basic heuristic)."""
    if not url or not slug:
        return False
    try:
        return slug.lower() in url.lower()
    except Exception:
        return False
