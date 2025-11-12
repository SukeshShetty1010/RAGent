# data/rawg_data.py
from typing import Optional, List, Dict, Any
from auth.rawg_client import RAWGClient

class RAWGData:
    """
    High-level data access for RAGent agent.
    Used as a Tool: agent calls methods to fetch grounded game data.
    """

    def __init__(self, client: Optional[RAWGClient] = None):
        self.client = client or RAWGClient()

    def search_and_rank_games(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search RAWG and return top_k games with essential metadata.
        Used by agent to discover relevant game IDs.
        """
        results = self.client.search_games(query, page_size=top_k)
        return [
            {
                "id": g.get("id"),
                "name": g.get("name"),
                "slug": g.get("slug"),
                "released": g.get("released"),
                "rating": g.get("rating"),
                "metacritic": g.get("metacritic"),
                "background_image": g.get("background_image")
            }
            for g in results
        ]

    def get_full_game_profile(self, game_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch and normalize full game profile with all related data.
        This is the 'ground truth' source for agent responses.
        """
        main = self.client.get_game_details(game_id)
        if not main:
            return None

        # Fetch related data
        additions = self.client.get_game_additions(game_id) or {}
        series = self.client.get_game_series(game_id) or {}
        achievements = self.client.get_achievements(game_id) or {}
        stores = self.client.get_stores(game_id) or {}

        # Normalize system requirements
        system_reqs = {}
        for p in main.get("platforms", []):
            platform = p.get("platform", {}).get("name")
            reqs = p.get("requirements", {})
            if platform and (reqs.get("minimum") or reqs.get("recommended")):
                system_reqs[platform] = {
                    "minimum": reqs.get("minimum"),
                    "recommended": reqs.get("recommended")
                }

        return {
            # Core
            "id": main.get("id"),
            "name": main.get("name"),
            "slug": main.get("slug"),
            "description": main.get("description_raw") or main.get("description") or "",
            "released": main.get("released"),
            "website": main.get("website"),
            "metacritic": main.get("metacritic"),
            "rating": main.get("rating"),
            "playtime": main.get("playtime"),
            "tba": main.get("tba", False),

            # Taxonomy
            "genres": [g.get("name") for g in main.get("genres", [])],
            "tags": [t.get("name") for t in main.get("tags", [])[:20]],
            "developers": [d.get("name") for d in main.get("developers", [])],
            "publishers": [p.get("name") for p in main.get("publishers", [])],
            "platforms": [p.get("platform", {}).get("name") for p in main.get("platforms", [])],

            # Ratings
            "esrb_rating": (main.get("esrb_rating") or {}).get("name"),

            # Relations
            "additions": [a.get("name") for a in additions.get("results", [])[:5]],
            "series": [s.get("name") for s in series.get("results", [])[:10]],

            # Achievements (sample)
            "achievements_sample": [
                {"name": a.get("name"), "description": a.get("description")}
                for a in achievements.get("results", [])[:5]
            ],

            # Stores
            "stores": [
                {"name": s.get("store", {}).get("name"), "url": s.get("url")}
                for s in stores.get("results", [])[:10]
                if s.get("store", {}).get("name")
            ],

            # Tech
            "system_requirements": system_reqs,
            "updated": main.get("updated"),
        }

    def get_game_by_name(self, name: str) -> Optional[Dict]:
        """
        Convenience: search by name and return best match full profile.
        """
        candidates = self.search_and_rank_games(name, top_k=1)
        if not candidates:
            return None
        return self.get_full_game_profile(candidates[0]["id"])