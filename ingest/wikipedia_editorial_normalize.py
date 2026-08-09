"""
Wikipedia Editorial Normalization Pipeline

Fetches, splits, and normalizes a Wikipedia article into the same
editorial object shape ingest/gamespot_editorial_normalize.py produces,
so EditorialChunker.process_game_editorial can consume it unchanged.
Strictly linked to a Canonical Game entity.

This module is intentionally limited to normalization only.
No chunking. No embeddings. No vector logic.
"""

from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid5

from data.wikipedia_data import fetch_wikipedia_article

# Matches MediaWiki plaintext section headings, e.g. "== Gameplay ==",
# "=== Reception ===". Captures the heading level (for the closing
# delimiter) and the heading text.
_SECTION_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

# Sections with no encyclopedic prose content worth chunking.
_BOILERPLATE_SECTIONS = {
    "references",
    "external links",
    "see also",
    "notes",
    "further reading",
    "bibliography",
}

_NS = UUID("12345678-1234-5678-1234-567812345678")


def _split_sections(extract: str) -> List[Dict[str, str]]:
    """
    Split a MediaWiki plaintext extract on "== Heading ==" markers into
    (title, body) sections. Text before the first heading becomes an
    "Overview" section. Boilerplate sections (references, external
    links, ...) are dropped.
    """
    matches = list(_SECTION_HEADING_RE.finditer(extract))
    sections: List[Dict[str, str]] = []

    lead = extract[: matches[0].start()].strip() if matches else extract.strip()
    if lead:
        sections.append({"title": "Overview", "body": lead})

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(extract)
        body = extract[start:end].strip()
        if not body or heading.lower() in _BOILERPLATE_SECTIONS:
            continue
        sections.append({"title": heading, "body": body})

    return sections


def fetch_and_prepare_wikipedia(
    game_name: str,
    canonical_game_uuid: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch, split, and normalize a Wikipedia article for `game_name`.

    Args:
        game_name: canonical title to resolve on Wikipedia — pass
            CanonicalIdentity.title (not the raw user query) so identity
            stays the single source of truth.
        canonical_game_uuid: UUID of the canonical Game entity (REQUIRED)

    Returns:
        Editorial payload dict (same shape as fetch_and_prepare_gamespot)
        or None if no article could be resolved.
    """
    if not canonical_game_uuid:
        raise ValueError("canonical_game_uuid is required (No Orphan Rule)")

    article = fetch_wikipedia_article(game_name)
    if not article:
        return None

    pageid = article.get("pageid")
    if not pageid:
        return None

    sections = _split_sections(article["extract"])
    if not sections:
        return None

    articles = [
        {
            "title": f"{article['title']} — {section['title']}",
            "date": None,
            "deck": None,
            "body": section["body"],
        }
        for section in sections
    ]

    editorial_object = {
        "metadata": {
            "id": pageid,
            "name": article["title"],
            "slug": None,
            "release_date": None,
        },
        "summary": {
            "deck": None,
            "description": sections[0]["body"][:500],
        },
        "reviews": {"average_score": None, "items": []},
        "articles": articles,
    }

    uuid_seed = f"wikipedia_{pageid}"
    wikipedia_uuid = str(uuid5(_NS, uuid_seed))

    return {
        "uuid": wikipedia_uuid,
        "class": "EditorialSource",
        "properties": {
            **editorial_object,
            "game_uuid": canonical_game_uuid,
            "source": "wikipedia",
        },
    }


# ------------------------------------------------------------------
# CLI test harness
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia Editorial Normalization")
    parser.add_argument("--game", "-g", required=True, help="Game title")
    parser.add_argument(
        "--uuid", required=True, help="Canonical Game UUID (dummy allowed for testing)"
    )

    args = parser.parse_args()

    result = fetch_and_prepare_wikipedia(game_name=args.game, canonical_game_uuid=args.uuid)

    print("\n=== Wikipedia Normalized Payload ===")
    print(result)
