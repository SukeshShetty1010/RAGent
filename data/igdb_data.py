# data/igdb_data.py
"""
High-level IGDBData — text-focused IGDB data retriever for RAGent.
Integrates RAWGData to first correct the game name before querying IGDB.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from auth.igdb_client import IGDBClient
from data.rawg_data import RAWGData


def _epoch_to_iso(ts: Optional[int]) -> Optional[str]:
    if not ts:
        return None
    try:
        return datetime.utcfromtimestamp(int(ts)).isoformat() + "Z"
    except Exception:
        return None


class IGDBData:
    """
    Fetches and normalizes IGDB game data (textual only) using the authenticated IGDBClient.
    """

    def __init__(self, client: Optional[IGDBClient] = None):
        self.client = client or IGDBClient()

    # ---------- Utility lookups ----------

    def _fetch_lookup(self, endpoint: str, ids: List[int], fields: str = "id,name") -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        ids_clause = ", ".join(str(i) for i in ids)
        body = f"where id = ({ids_clause}); fields {fields}; limit {len(ids)};"
        items = self.client.post(endpoint, body)
        return {i["id"]: i for i in items if "id" in i}

    # ---------- Core methods ----------

    def _normalize_game(self, raw: Dict[str, Any], lookups: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
        def map_names(key: str, ids: List[int]) -> List[str]:
            if not ids:
                return []
            lookup = lookups.get(key, {})
            out = []
            for i in ids:
                rec = lookup.get(i)
                if rec and rec.get("name"):
                    out.append(rec["name"])
            return out

        return {
            "id": raw.get("id"),
            "name": raw.get("name"),
            "slug": raw.get("slug"),
            "description": raw.get("description") or raw.get("storyline") or raw.get("summary"),
            "summary": raw.get("summary"),
            "storyline": raw.get("storyline"),
            "rating": raw.get("rating"),
            "total_rating": raw.get("total_rating"),
            "aggregated_rating": raw.get("aggregated_rating"),
            "rating_count": raw.get("rating_count"),
            "release_date": _epoch_to_iso(raw.get("first_release_date")),
            "updated_at": _epoch_to_iso(raw.get("updated_at")),
            "version_title": raw.get("version_title"),
            "genres": map_names("genres", raw.get("genres", [])),
            "themes": map_names("themes", raw.get("themes", [])),
            "platforms": map_names("platforms", raw.get("platforms", [])),
            "franchises": map_names("franchises", raw.get("franchises", [])),
            "collections": map_names("collections", raw.get("collections", [])),
            "involved_companies": map_names("involved_companies", raw.get("involved_companies", [])),
            "game_modes": raw.get("game_modes", []),
            "player_perspectives": raw.get("player_perspectives", []),
            "keywords": raw.get("keywords", []),
            "similar_games": map_names("games", raw.get("similar_games", [])),
            "source": "IGDB"
        }

    def get_game_by_name(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Use RAWGData to correct the name, then query IGDB and return top-1 normalized game.
        """
        rawg = RAWGData()
        corrected = rawg.search_and_rank_games(query, top_k=1)
        if not corrected:
            print(f"[IGDBData] RAWG returned no matches for '{query}'.")
            return None

        correct_name = corrected[0]["name"]
        print(f"[IGDBData] Corrected search term via RAWG → '{correct_name}'")

        # Search IGDB
        body = f'search "{correct_name}"; fields *; limit 1;'
        results = self.client.post("games", body)
        if not results:
            print(f"[IGDBData] IGDB returned no data for '{correct_name}'.")
            return None

        game = results[0]

        # Collect reference IDs
        ref_fields = ["genres", "themes", "platforms", "involved_companies", "franchises", "collections", "games"]
        id_buckets = {k: set(game.get(k, [])) for k in ref_fields}

        lookups = {}
        for key, ids in id_buckets.items():
            if not ids:
                lookups[key] = {}
                continue
            endpoint = key
            fields = "id,name" if key != "involved_companies" else "id,company"
            lookups[key] = self._fetch_lookup(endpoint, list(ids), fields)

        # Resolve company names
        if lookups.get("involved_companies"):
            company_ids = [
                rec.get("company") for rec in lookups["involved_companies"].values()
                if isinstance(rec.get("company"), int)
            ]
            if company_ids:
                company_map = self._fetch_lookup("companies", company_ids, "id,name")
                for iid, rec in lookups["involved_companies"].items():
                    cid = rec.get("company")
                    if cid and company_map.get(cid):
                        rec["name"] = company_map[cid]["name"]

        return self._normalize_game(game, lookups)
