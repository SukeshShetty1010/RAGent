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
from typing import Any, Dict, Optional, Tuple
from uuid import UUID, uuid5

GAME_NAMESPACE_UUID = UUID("12345678-1234-5678-1234-567812345678")

# Strips a trailing parenthetical year qualifier (e.g. the TOP_100_GAMES
# entries "Resident Evil 4 (2023)", "Dead Space (2023)") so the real
# release year is appended exactly once instead of twice.
_TRAILING_PAREN_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")
_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def slug(title: str) -> str:
    """Normalize a title into a URL-safe slug for use in unified_game_id."""
    stripped = _TRAILING_PAREN_YEAR.sub("", title).strip().lower()
    slugged = _NON_SLUG_CHARS.sub("-", stripped).strip("-")
    return slugged


def make_unified_game_id(title: str, release_year: Optional[int]) -> str:
    base = slug(title)
    if release_year is not None:
        return f"{base}-{release_year}"
    return base


def make_game_uuid(unified_game_id: str) -> str:
    return str(uuid5(GAME_NAMESPACE_UUID, unified_game_id))


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

    try:
        result = fetch_igdb_game_data(game_name, strip_visual=True, limit=1)
    except Exception:
        return None

    records = result.get("clean") or []
    if not records:
        return None

    record = records[0]
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
