"""
ingest/identity_resolver.py

Provider-independent canonical identity resolution for the Game anchor
(Stage 1 of the ingestion pipeline).

Previously Stage 1 rooted the Game UUID in RAWG's integer `game_id`,
which made the entire pipeline a hard dependency on a single upstream
API (see upsert/upsert_canonical_game.py's prior version). This module
resolves identity from an ordered list of providers and derives
`unified_game_id` / `game_uuid` from title+year instead of any one
vendor's primary key — so identity is stable (byte-identical UUIDs)
regardless of which provider answers, or whether a provider is down.

Provider order: IGDB, then RAWG. Each adapter fails soft (returns None
rather than raising) so a provider outage just falls through to the
next one; resolution only fails when every provider is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid5

GAME_NAMESPACE_UUID = UUID("12345678-1234-5678-1234-567812345678")

# Strips a trailing parenthetical year qualifier (e.g. the TOP_100_GAMES
# entries "Resident Evil 4 (2023)", "Dead Space (2023)") so the real
# release year is appended exactly once instead of twice.
_TRAILING_PAREN_YEAR = re.compile(r"\s*\((\d{4})\)\s*$")
_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def slug(title: str) -> str:
    """Normalize a title into a URL-safe slug for use in unified_game_id."""
    stripped = _TRAILING_PAREN_YEAR.sub("", title).strip().lower()
    slugged = _NON_SLUG_CHARS.sub("-", stripped).strip("-")
    return slugged


def split_trailing_year(title: str) -> Tuple[str, Optional[int]]:
    """
    Split a trailing "(YYYY)" qualifier off a title, e.g.
    "Resident Evil 4 (2023)" -> ("Resident Evil 4", 2023).

    IGDB's `search` endpoint returns zero hits on a literal parenthetical
    suffix, so callers must search on the stripped title — the year is
    kept separately to disambiguate same-titled remakes/originals
    (e.g. picking the 2023 Dead Space remake over the 2008 original).
    """
    match = _TRAILING_PAREN_YEAR.search(title)
    if not match:
        return title, None
    return _TRAILING_PAREN_YEAR.sub("", title).strip(), int(match.group(1))


def make_unified_game_id(title: str, release_year: Optional[int]) -> str:
    base = slug(title)
    if release_year is not None:
        return f"{base}-{release_year}"
    return base


def make_game_uuid(unified_game_id: str) -> str:
    return str(uuid5(GAME_NAMESPACE_UUID, unified_game_id))


def _igdb_release_year(record: Dict[str, Any]) -> Optional[int]:
    ts = record.get("first_release_date")
    if isinstance(ts, (int, float)):
        from datetime import datetime, timezone

        return datetime.fromtimestamp(ts, tz=timezone.utc).year
    return None


def _igdb_release_sort_key(record: Dict[str, Any]) -> float:
    ts = record.get("first_release_date")
    return float(ts) if isinstance(ts, (int, float)) else float("inf")


def select_best_igdb_match(
    records: List[Dict[str, Any]],
    query: str,
    year_hint: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Exact-match preference ladder for IGDB search results, mirroring
    RAWG's `prefer_exact_match` (data/rawg_data.py:133-139).

    IGDB's `search` is fuzzy — a plain top-1 pick can select the wrong
    game (e.g. "Elden Ring" -> "Elden Ring Nightreign"). Tries, in order:
      1. normalized exact title match (slug equality); if several share
         the title (base game vs. remake/edition), `year_hint` (from
         `split_trailing_year`) picks the one matching that release year
      2. candidate whose slug starts with the query slug, same year_hint
         tiebreak, else earliest first_release_date
      3. first record (previous behavior)
    """
    if not records:
        return None

    query_slug = slug(query)

    def _pick(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        if year_hint is not None:
            year_matches = [r for r in pool if _igdb_release_year(r) == year_hint]
            if year_matches:
                return sorted(year_matches, key=_igdb_release_sort_key)[0]
        return sorted(pool, key=_igdb_release_sort_key)[0]

    exact_matches = [
        record for record in records if slug(record.get("name") or "") == query_slug
    ]
    if exact_matches:
        return _pick(exact_matches)

    prefix_matches = [
        record
        for record in records
        if slug(record.get("name") or "").startswith(query_slug)
    ]
    if prefix_matches:
        return _pick(prefix_matches)

    return records[0]


@dataclass
class CanonicalIdentity:
    unified_game_id: str
    game_uuid: str
    title: str
    release_year: Optional[int]
    identity_source: str
    source_ids: Dict[str, int] = field(default_factory=dict)


class IdentityResolutionError(RuntimeError):
    """Raised when no configured provider can resolve a game's identity."""


# ---------------------------------------------------------------------
# Provider adapters.
#
# Each adapter takes a game name and returns (title, release_year,
# source_id) or None if it can't resolve. Adapters must not raise for
# "not found" or transport failures — those are fail-soft None returns
# so the resolver can fall through to the next provider.
# ---------------------------------------------------------------------

_ProviderResult = Optional[Tuple[str, Optional[int], Optional[int]]]


def _igdb_adapter(game_name: str) -> _ProviderResult:
    from data.igdb_data import fetch_igdb_game_data

    clean_query, year_hint = split_trailing_year(game_name)

    try:
        result = fetch_igdb_game_data(clean_query, strip_visual=True, limit=10)
    except Exception:
        return None

    records = result.get("clean") or []
    if not records:
        return None

    record = select_best_igdb_match(records, clean_query, year_hint=year_hint)
    if record is None:
        return None
    title = record.get("name") or result.get("resolved_name")
    if not title:
        return None

    release_year: Optional[int] = None
    ts = record.get("first_release_date")
    if isinstance(ts, (int, float)):
        from datetime import datetime, timezone

        release_year = datetime.fromtimestamp(ts, tz=timezone.utc).year

    igdb_id = record.get("id")
    return title, release_year, int(igdb_id) if igdb_id is not None else None


def _rawg_adapter(game_name: str) -> _ProviderResult:
    from ingest.rawg_identity_ingest import fetch_and_prepare_identity

    try:
        game_obj = fetch_and_prepare_identity(game_name)
    except Exception:
        return None

    title = game_obj.get("title")
    if not title:
        return None

    return title, game_obj.get("release_year"), game_obj.get("game_id")


_PROVIDERS: Tuple[Tuple[str, Any], ...] = (
    ("igdb", _igdb_adapter),
    ("rawg", _rawg_adapter),
)


def resolve_identity(game_name: str) -> CanonicalIdentity:
    """
    Resolve a provider-independent canonical identity for `game_name`.

    Walks providers in order (IGDB, then RAWG), returning on first
    success. Because the UUID is seeded from title+year rather than a
    vendor ID, a later fallback to a different provider — or RAWG
    rejoining after an outage — reproduces the same game_uuid.

    Raises IdentityResolutionError if every provider fails.
    """
    if not game_name or not game_name.strip():
        raise IdentityResolutionError("game_name must be non-empty")

    for source_name, adapter in _PROVIDERS:
        resolved = adapter(game_name)
        if resolved is None:
            continue

        title, release_year, source_id = resolved
        unified_game_id = make_unified_game_id(title, release_year)

        return CanonicalIdentity(
            unified_game_id=unified_game_id,
            game_uuid=make_game_uuid(unified_game_id),
            title=title,
            release_year=release_year,
            identity_source=source_name,
            source_ids={source_name: source_id} if source_id is not None else {},
        )

    raise IdentityResolutionError(
        f"No provider could resolve identity for {game_name!r}"
    )
