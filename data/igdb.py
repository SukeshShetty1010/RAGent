from typing import List, Dict, Any, Optional
from api.igdb_client import igdb_request

class IGDBTool:
    def search_games(self, query: str, limit: int = 10, fields: str = "*") -> List[Dict[str, Any]]:
        q = query.replace("’", "'").strip()

        body = f'search "{q}"; fields {fields}; limit {limit};'
        results = igdb_request("games", body)
        if results:
            return results

        body = f'fields {fields}; where name ~ *"{q}"*; limit {limit};'
        results = igdb_request("games", body)
        if results:
            return results

        last = q.split()[-1]
        body = f'fields {fields}; where name ~ *"{last}"*; limit {limit};'
        results = igdb_request("games", body)
        return results or []

    def get_recent_games(self, limit: int = 20) -> List[Dict[str, Any]]:
        body = f"""
        fields name, summary, first_release_date, genres.name, platforms.name;
        where first_release_date != null & category = 0;
        sort first_release_date desc;
        limit {limit};
        """
        return igdb_request("games", body.strip())

    def get_game_by_id(self, game_id: int) -> Optional[Dict[str, Any]]:
        body = f"fields *; where id = {game_id};"
        results = igdb_request("games", body)
        return results[0] if results else None

    def get_expansions_and_dlcs(self, game_id: int) -> List[Dict[str, Any]]:
        """OFFICIAL IGDB METHOD: Fetch from dlcs + expansions arrays"""
        base = self.get_game_by_id(game_id)
        if not base:
            return []

        all_ids = []
        if "dlcs" in base and base["dlcs"]:
            all_ids.extend(base["dlcs"])
        if "expansions" in base and base["expansions"]:
            all_ids.extend(base["expansions"])

        if not all_ids:
            return []

        ids_str = ",".join(str(i) for i in all_ids)
        body = f"""
        fields name, summary, first_release_date, category, cover.url;
        where id = ({ids_str});
        limit 100;
        """
        return igdb_request("games", body.strip())

    def fetch_both(self, query: str = "", limit: int = 10) -> Dict[str, Any]:
        recent = self.get_recent_games(limit // 2)
        searched = self.search_games(query, limit // 2) if query else []
        return {"recent_games": recent, "searched_games": searched}

igdb_tool = IGDBTool()