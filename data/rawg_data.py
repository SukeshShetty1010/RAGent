"""
data/rawg_data.py

Utility function to fetch game data from RAWG API and return either the full JSON
or a 'stripped' version with visual/media fields removed (for ingestion into RAG).

Usage:
from data.rawg_data import fetch_rawg_game_data
game = fetch_rawg_game_data("elden ring")
"""

import os
import threading
import time
import requests
import json
import re
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from copy import deepcopy

load_dotenv()

RAWG_BASE = "https://api.rawg.io/api"
DEFAULT_TIMEOUT = 5

# Hard kill switch — set RAWG_ENABLED=false in .env to skip RAWG entirely.
RAWG_ENABLED = os.getenv("RAWG_ENABLED", "true").strip().lower() not in (
    "false",
    "0",
    "no",
)

# Consecutive-failure circuit breaker. RAWG going dark otherwise stalls
# every call site (fetch_rawg_game_data, igdb_data._resolve_via_rawg,
# gamespot_data.get_correct_game_name_from_rawg) behind its own timeout
# — ~30-60s of dead wait per game across ~100 games. Shared process-wide
# so a failure recorded by any call site trips the breaker for all three.
_RAWG_FAILURE_THRESHOLD = 3
_rawg_lock = threading.Lock()
_rawg_consecutive_failures = 0


def rawg_available() -> bool:
    """
    False if RAWG_ENABLED is off, or after 3 consecutive network
    failures in this process. Self-heals: a later success resets the
    counter, so RAWG rejoins automatically once the host recovers.
    """
    if not RAWG_ENABLED:
        return False
    with _rawg_lock:
        return _rawg_consecutive_failures < _RAWG_FAILURE_THRESHOLD


def record_rawg_success() -> None:
    global _rawg_consecutive_failures
    with _rawg_lock:
        _rawg_consecutive_failures = 0


def record_rawg_failure() -> None:
    global _rawg_consecutive_failures
    with _rawg_lock:
        _rawg_consecutive_failures += 1


_KEY_PARAM_RE = re.compile(r"([?&]key=)[^&\s]+")


def _redact(text: str) -> str:
    """Strip the RAWG API key out of error text before it reaches logs."""
    return _KEY_PARAM_RE.sub(r"\1***", text)


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


def _safe_get(
    url: str, params: Dict[str, Any], timeout: int = DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        record_rawg_failure()
        raise RawgError(_redact(f"Network error calling RAWG: {e}"))

    if resp.status_code == 404:
        record_rawg_success()
        raise RawgError("RAWG resource not found (404)")

    try:
        data = resp.json()
    except ValueError:
        record_rawg_failure()
        raise RawgError("RAWG returned non-JSON response")

    if resp.status_code >= 400:
        msg = data.get("detail") if isinstance(data, dict) else None
        record_rawg_failure()
        raise RawgError(_redact(f"RAWG API error {resp.status_code}: {msg or resp.text}"))

    record_rawg_success()
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

    if not rawg_available():
        raise RawgError("RAWG unavailable (disabled or circuit open)")

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