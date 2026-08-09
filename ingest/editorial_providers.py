"""
ingest/editorial_providers.py

Editorial source provider registry (Stage 4 of the ingestion pipeline).

Each provider returns the editorial object shape
EditorialChunker.process_game_editorial consumes, wrapped in a
container envelope: {"uuid", "class", "properties": {..., "game_uuid",
"source"}} — or None if that provider has nothing for this game.

Every provider is wrapped so a raised exception never kills the others;
a game succeeds if ANY provider yields content. GameSpot stays first in
the list (existing primary source, auto-reactivates once its Cloudflare
block lifts); Wikipedia and Steam are additive, not replacements.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _gamespot_provider(game_name: str, canonical_game_uuid: str) -> Optional[Dict[str, Any]]:
    from ingest.gamespot_editorial_normalize import fetch_and_prepare_gamespot

    return fetch_and_prepare_gamespot(game_name, canonical_game_uuid)


def _wikipedia_provider(game_name: str, canonical_game_uuid: str) -> Optional[Dict[str, Any]]:
    from ingest.wikipedia_editorial_normalize import fetch_and_prepare_wikipedia

    return fetch_and_prepare_wikipedia(game_name, canonical_game_uuid)


def _steam_provider(game_name: str, canonical_game_uuid: str) -> Optional[Dict[str, Any]]:
    from ingest.steam_editorial_normalize import fetch_and_prepare_steam

    return fetch_and_prepare_steam(game_name, canonical_game_uuid)


_ProviderFn = Callable[[str, str], Optional[Dict[str, Any]]]

PROVIDERS: Tuple[Tuple[str, _ProviderFn], ...] = (
    ("gamespot", _gamespot_provider),
    ("wikipedia", _wikipedia_provider),
    ("steam", _steam_provider),
)


def fetch_all_editorial_containers(
    game_name: str, canonical_game_uuid: str
) -> List[Dict[str, Any]]:
    """
    Run every registered provider for `game_name`, fail-soft.

    Returns the list of container payloads that succeeded (possibly
    empty — callers must treat an empty list as "no editorial content
    this run", not an error; the honesty gate depends on that
    distinction propagating up rather than being masked by a crash).
    """
    containers: List[Dict[str, Any]] = []

    for source_name, provider in PROVIDERS:
        try:
            container = provider(game_name, canonical_game_uuid)
        except Exception:
            logger.warning(
                "Editorial provider '%s' raised for '%s' — skipping.",
                source_name,
                game_name,
                exc_info=True,
            )
            continue

        if container:
            containers.append(container)
        else:
            logger.info("Editorial provider '%s' had no data for '%s'.", source_name, game_name)

    return containers
