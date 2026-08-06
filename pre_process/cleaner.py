"""
cleaner.py
==========

Unified cleaning utilities for game metadata ingestion.

This module contains:
- RAWGCleaner      : Cleans RAWG game data (UNCHANGED)
- IGDBCleaner      : Cleans IGDB game data into relational tables (UNCHANGED)
- GameSpotCleaner  : Cleans GameSpot game data into a strict editorial schema

Design principles:
- No global execution or side effects
- Deterministic, schema-preserving transformations
- Stateless class design
- Type hints and defensive checks
"""

import logging
import re
from html import unescape
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ======================================================================
# RAWG CLEANER (UNCHANGED — DO NOT MODIFY)
# ======================================================================
class RAWGCleaner:
    """
    Cleaner for RAWG game JSON payloads.
    """

    def clean(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(data, dict):
            logger.warning("RAWGCleaner.clean received non-dict input")
            return None

        try:
            return self._transform_rawg(data)
        except Exception as exc:
            logger.exception("RAWG cleaning failed: %s", exc)
            raise

    def _transform_rawg(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        game_id = raw.get("id")
        title = raw.get("name")
        description = self.strip_html(
            raw.get("description") or raw.get("description_raw")
        )

        release_date = raw.get("released")
        release_year = None
        if release_date:
            try:
                release_year = int(release_date[:4])
            except Exception:
                release_year = None

        parent_lookup = self.build_platform_family_lookup(
            raw.get("parent_platforms") or []
        )

        platforms_out: List[Dict[str, Any]] = []
        platforms = raw.get("platforms") or []

        if not platforms:
            for pname in parent_lookup.keys():
                platforms_out.append(
                    {
                        "platform_name": pname,
                        "platform_family": pname,
                        "release_date": None,
                        "requirements_minimum": None,
                        "requirements_recommended": None,
                    }
                )
        else:
            for p in platforms:
                platform_info = p.get("platform") or {}
                requirements = p.get("requirements") or {}

                platform_name = platform_info.get("name")
                platforms_out.append(
                    {
                        "platform_name": platform_name,
                        "platform_family": parent_lookup.get(platform_name),
                        "release_date": p.get("released_at"),
                        "requirements_minimum": requirements.get("minimum"),
                        "requirements_recommended": requirements.get("recommended"),
                    }
                )

        genres = self.dedupe_strings(
            [g.get("name") for g in (raw.get("genres") or []) if g.get("name")]
        )

        developers = self.dedupe_strings(
            [d.get("name") for d in (raw.get("developers") or []) if d.get("name")]
        )

        publishers = self.dedupe_strings(
            [p.get("name") for p in (raw.get("publishers") or []) if p.get("name")]
        )

        ratings_out = {
            "average_rating": raw.get("rating"),
            "rating_top": raw.get("rating_top"),
            "metacritic": raw.get("metacritic"),
            "distribution": [],
            "metacritic_by_platform": [],
        }

        for r in raw.get("ratings") or []:
            ratings_out["distribution"].append(
                {
                    "label": r.get("title"),
                    "percent": float(r.get("percent"))
                    if r.get("percent") is not None
                    else None,
                    "count": int(r.get("count"))
                    if r.get("count") is not None
                    else None,
                }
            )

        for mp in raw.get("metacritic_platforms") or []:
            plat = mp.get("platform") or {}
            ratings_out["metacritic_by_platform"].append(
                {
                    "platform": plat.get("name"),
                    "score": mp.get("metascore"),
                }
            )

        tags = [
            t.get("name")
            for t in (raw.get("tags") or [])
            if t.get("language") == "eng" and t.get("name")
        ]

        engagement = {
            "added_by_status": raw.get("added_by_status") or {},
            "reactions": raw.get("reactions") or {},
        }

        age_rating = {"esrb": raw.get("esrb_rating")}

        source = {
            "rawg_url": raw.get("website"),
            "last_updated": raw.get("updated"),
        }

        return {
            "game_id": game_id,
            "title": title,
            "description": description,
            "release_date": release_date,
            "release_year": release_year,
            "platforms": platforms_out,
            "genres": genres,
            "developers": developers,
            "publishers": publishers,
            "ratings": ratings_out,
            "tags": tags,
            "age_rating": age_rating,
            "engagement": engagement,
            "source": source,
        }

    @staticmethod
    def strip_html(text: Optional[str]) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        text = unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None

    @staticmethod
    def dedupe_strings(values: List[str]) -> List[str]:
        seen = set()
        result = []
        for v in values:
            key = v.lower()
            if key not in seen:
                seen.add(key)
                result.append(v)
        return result

    @staticmethod
    def build_platform_family_lookup(
        parent_platforms: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for p in parent_platforms or []:
            platform = p.get("platform") or {}
            name = platform.get("name")
            if name:
                lookup[name] = name
        return lookup


# ======================================================================
# IGDB CLEANER (UNCHANGED — DO NOT MODIFY)
# ======================================================================
class IGDBCleaner:
    """
    Cleaner for IGDB game records.
    """

    def clean(self, items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        return self.clean_batch(items)

    def clean_batch(self, items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        if not isinstance(items, list):
            raise ValueError("IGDBCleaner.clean_batch expects a list")

        extracted_items: List[Dict[str, Any]] = []

        for item in items:
            if isinstance(item, dict) and "clean" in item and isinstance(item["clean"], list):
                extracted_items.extend(item["clean"])
            elif isinstance(item, dict):
                extracted_items.append(item)

        return self._build_relational_structure(extracted_items)

    def _build_relational_structure(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:

        output = {
            "Main_Game_Table": [],
            "Expansion_Table": [],
            "Edition_Table": [],
            "Bundle_Table": [],
            "Update_Table": [],
            "Standalone_Experience_Table": [],
            "Unclassified": [],
        }

        for item in items:
            table = self._classify_entity(item)
            transformed = self._transform_item(item)
            output[table].append(transformed)

        return output

    @staticmethod
    def _classify_entity(item: Dict[str, Any]) -> str:
        game_type = item.get("game_type")
        parent_game = item.get("parent_game")
        version_parent = item.get("version_parent")

        if game_type == 0 and not parent_game and not version_parent:
            return "Main_Game_Table"
        if game_type == 0 and version_parent:
            return "Edition_Table"
        if game_type in (1, 2) and parent_game:
            return "Expansion_Table"
        if game_type == 14 and parent_game:
            return "Update_Table"
        if game_type == 3:
            return "Bundle_Table"
        if game_type == 4:
            return "Standalone_Experience_Table"

        return "Unclassified"

    def _transform_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": item.get("id"),
            "name": item.get("name"),
            "parent_game": item.get("parent_game"),
            "version_parent": item.get("version_parent"),
            "version_title": item.get("version_title"),
            "game_type": item.get("game_type"),
            "first_release_date": self._unix_to_iso(item.get("first_release_date")),
            "created_at": self._unix_to_iso(item.get("created_at")),
            "updated_at": self._unix_to_iso(item.get("updated_at")),
            "summary": self._normalize_text(item.get("summary")),
            "storyline": self._normalize_text(item.get("storyline")),
            "aggregated_rating": item.get("aggregated_rating"),
            "total_rating": item.get("total_rating"),
            "checksum": item.get("checksum"),
            "genres": item.get("genres", []),
            "platforms": item.get("platforms", []),
            "themes": item.get("themes", []),
            "keywords": item.get("keywords", []),
            "tags": item.get("tags", []),
            "involved_companies": item.get("involved_companies", []),
            "franchises": item.get("franchises", []),
            "collections": item.get("collections", []),
        }

    @staticmethod
    def _unix_to_iso(ts: Optional[int]) -> Optional[str]:
        if not ts:
            return None
        try:
            return datetime.utcfromtimestamp(ts).isoformat() + "Z"
        except Exception:
            return None

    @staticmethod
    def _normalize_text(text: Optional[str]) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        return "\n".join(line.strip() for line in text.strip().splitlines())


# ======================================================================
# GAMESPOT CLEANER (NEW — REFACTORED FROM clean_GameSpot.py)
# ======================================================================
class GameSpotCleaner:
    """
    Cleaner for GameSpot game payloads.

    Expects the raw GameSpot dictionary at root level and produces
    a strict editorial schema containing:
    - metadata
    - summary
    - reviews
    - articles
    """

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def clean(self, source: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(source, dict):
            raise ValueError("GameSpotCleaner.clean expects a dictionary")

        games = self._safe_list(source.get("games"))
        first = games[0] if games else {}

        game = first.get("game", {})
        related = first.get("related", {})

        reviews_src = self._safe_list(related.get("reviews"))
        articles_src = self._safe_list(related.get("articles"))

        review_scores = [r.get("score") for r in reviews_src]

        return {
            "metadata": {
                "id": game.get("id"),
                "name": game.get("name"),
                "slug": game.get("slug"),
                "release_date": self._iso_date(
                    game.get("original_release_date") or game.get("release_date")
                ),
            },
            "summary": {
                "deck": game.get("deck"),
                "description": game.get("description"),
            },
            "reviews": {
                "average_score": self._avg(review_scores),
                "items": [
                    {
                        "title": r.get("title"),
                        "score": r.get("score"),
                        "date": self._iso_date(r.get("publish_date")),
                        "body": r.get("review_text") or r.get("body"),
                    }
                    for r in reviews_src
                ],
            },
            "articles": [
                {
                    "title": a.get("title"),
                    "date": self._iso_date(a.get("publish_date")),
                    "deck": a.get("deck"),
                    "body": a.get("body"),
                }
                for a in articles_src
            ],
        }

    # --------------------------------------------------
    # Helpers (GameSpot-only)
    # --------------------------------------------------
    @staticmethod
    def _iso_date(val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(val[: len(fmt)], fmt).date().isoformat()
            except Exception:
                continue
        return None

    @staticmethod
    def _safe_list(x: Any) -> List[Any]:
        return x if isinstance(x, list) else []

    @staticmethod
    def _avg(nums: List[float]) -> Optional[float]:
        nums = [n for n in nums if isinstance(n, (int, float))]
        return round(sum(nums) / len(nums), 2) if nums else None
