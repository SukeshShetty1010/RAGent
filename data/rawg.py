from typing import Dict, Any
from api.rawg_client import search_games, get_game_details

class RawgTool:
    """
    Tool for fetching data from RAWG.io API.
    Mirrors structure of IGDBTool for consistency in the agent.
    """
    
    def fetch_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for games on RAWG.
        
        Args:
            query (str): Search query (e.g., game title or keyword).
            limit (int): Max results to return (default: 10).
        
        Returns:
            dict: {"results": list of game dicts}
        """
        return {"results": search_games(query, page_size=limit)}

    def fetch_details(self, game_id: int) -> Dict[str, Any]:
        """
        Get detailed info for a specific game by ID.
        
        Args:
            game_id (int): RAWG game ID.
        
        Returns:
            dict: Game details.
        """
        return get_game_details(game_id) or {}

    def fetch_both(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Search for games and fetch details for the top result.
        Useful for quick lookups, similar to rough.py behavior.
        
        Args:
            query (str): Search query.
            limit (int): Max search results (default: 5).
        
        Returns:
            dict: {"search_results": list, "top_details": dict}
        """
        search_results = search_games(query, page_size=limit)
        details = {}
        if search_results:
            first_id = search_results[0].get("id")
            if first_id:
                details = self.fetch_details(first_id)
        return {
            "search_results": search_results,
            "top_details": details
        }