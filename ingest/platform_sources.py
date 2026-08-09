"""
ingest/platform_sources.py

Provider-independent PlatformSpec data (RAWG substitute): IGDB supplies
platform names, Steam supplies PC hardware requirements (the one field
IGDB's API doesn't expose). Builds the same `cleaned_data` shape
RAWGCleaner produces — {"platforms": [{"platform_name",
"platform_family", "release_date", "requirements_minimum",
"requirements_recommended"}, ...]} — so
ingest/platformspec_ingest.py::generate_platform_payloads and its
downstream upsert path stay unchanged regardless of source.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# IGDB platform id -> name, cached per process. Platform IDs are a small,
# stable set (a few hundred), so caching avoids a lookup call per game.
_platform_name_cache: Dict[int, str] = {}


def _fetch_igdb_platform_names(platform_ids: List[int]) -> Dict[int, str]:
    missing = [pid for pid in platform_ids if pid not in _platform_name_cache]
    if missing:
        from data.igdb_data import IGDB_PLATFORMS_URL, _get_twitch_token, _igdb_post

        try:
            token = _get_twitch_token()
            ids_clause = ",".join(str(pid) for pid in missing)
            records = _igdb_post(
                f"fields name; where id = ({ids_clause}); limit {len(missing)};",
                token,
                url=IGDB_PLATFORMS_URL,
            )
            for record in records:
                pid = record.get("id")
                name = record.get("name")
                if pid is not None and name:
                    _platform_name_cache[pid] = name
        except Exception:
            logger.warning(
                "IGDB platform name lookup failed; falling back to raw IDs."
            )

    return {pid: _platform_name_cache.get(pid, f"Platform {pid}") for pid in platform_ids}


def _igdb_platform_entries(game_name: str) -> List[Dict[str, Any]]:
    from data.igdb_data import fetch_igdb_game_data
    from ingest.identity_resolver import select_best_igdb_match, split_trailing_year

    clean_query, year_hint = split_trailing_year(game_name)

    try:
        result = fetch_igdb_game_data(clean_query, strip_visual=True, limit=10)
    except Exception:
        return []

    records = result.get("clean") or []
    record = select_best_igdb_match(records, clean_query, year_hint=year_hint)
    if not record:
        return []

    platform_ids = record.get("platforms") or []
    if not platform_ids:
        return []

    names = _fetch_igdb_platform_names(platform_ids)

    return [
        {
            "platform_name": names.get(pid, f"Platform {pid}"),
            "platform_family": names.get(pid, f"Platform {pid}"),
            "release_date": None,
            "requirements_minimum": None,
            "requirements_recommended": None,
        }
        for pid in platform_ids
    ]


def _apply_steam_pc_requirements(platforms: List[Dict[str, Any]], game_name: str) -> None:
    """
    Mutates `platforms` in place: fills PC requirements_minimum /
    requirements_recommended from Steam's pc_requirements. IGDB has no
    hardware-requirements data — Steam is the only substitute for it.
    """
    from data.steam_data import fetch_steam_game_data

    try:
        steam_data = fetch_steam_game_data(game_name)
    except Exception:
        steam_data = None

    if not steam_data:
        return

    pc_req = steam_data.get("pc_requirements") or {}
    minimum = pc_req.get("minimum")
    recommended = pc_req.get("recommended")
    if not minimum and not recommended:
        return

    release_date = (steam_data.get("release_date") or {}).get("date")

    for entry in platforms:
        if (entry.get("platform_name") or "").strip().lower() in (
            "pc",
            "pc (microsoft windows)",
            "windows",
            "pc dos",
        ):
            entry["requirements_minimum"] = minimum
            entry["requirements_recommended"] = recommended
            if release_date and not entry.get("release_date"):
                entry["release_date"] = release_date
            return

    # IGDB didn't surface a PC platform entry — add one from Steam directly.
    platforms.append(
        {
            "platform_name": "PC",
            "platform_family": "PC",
            "release_date": release_date,
            "requirements_minimum": minimum,
            "requirements_recommended": recommended,
        }
    )


def build_platform_cleaned_data(game_name: str) -> Optional[Dict[str, Any]]:
    """
    Provider-independent substitute for RAWGCleaner's output shape.

    Returns None if neither IGDB nor Steam yield any platform data —
    callers should treat that as "no platform specs this run", not an
    error (mirrors RAWGCleaner.clean's None-on-nothing-found contract).
    """
    platforms = _igdb_platform_entries(game_name)
    _apply_steam_pc_requirements(platforms, game_name)

    if not platforms:
        return None

    return {"platforms": platforms}
