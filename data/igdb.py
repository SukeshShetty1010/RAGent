# tools/igdb.py
from typing import Dict, Any, Optional
from api.igdb_client import igdb_request  # Import the IGDB request function

class IGDBTool:
    """
    A tool for fetching games data from the IGDB API.
    This tool can perform general game searches or retrieve recent games, with optional limits.
    """

    def __init__(self):
        """
        Initialize the IGDBTool. Authentication is handled internally via igdb_client.
        """
        pass

    def run(self, query: str | None = None, recent_games: bool = False, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Run the IGDB fetch operation.
        
        Args:
            query (str, optional): The search query for games. Required if not fetching recent games.
            recent_games (bool): If True, fetch recent games instead of searching. Defaults to False.
            limit (int): Number of results to return. Defaults to 10.
        
        Returns:
            list[Dict[str, Any]]: List of game data from the API.
        
        Raises:
            ValueError: If query is missing when not fetching recent games.
        """
        if recent_games:
            return self.get_recent_games(limit=limit)
        else:
            if not query:
                raise ValueError("Query is required for game search.")
            return self.search_games(query=query, limit=limit)

    def get_recent_games(self, limit: int = 20) -> list[Dict[str, Any]]:
        """
        Fetch recently released games, sorted by first_release_date descending.
        
        Args:
            limit (int): Number of results to return. Defaults to 20.
        
        Returns:
            list[Dict[str, Any]]: List of recent game data.
        """
        query = (
            f"fields id, name, summary, first_release_date, genres.name, platforms.name;"
            f"sort first_release_date desc; limit {limit};"
        )
        return igdb_request("games", query)

    def search_games(self, query: str, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Search for games by keyword.
        
        Args:
            query (str): The search query.
            limit (int): Number of results to return. Defaults to 10.
        
        Returns:
            list[Dict[str, Any]]: List of matching game data.
        """
        search_query = (
            f'search "{query}"; '
            f"fields id, name, summary, first_release_date, genres.name, platforms.name; "
            f"limit {limit};"
        )
        return igdb_request("games", search_query)

    def fetch_both(self, query: str, limit: int = 10) -> Dict[str, list[Dict[str, Any]]]:
        """
        Fetch both recent games and search results without storing to a file.
        
        Args:
            query (str): The search query for games.
            limit (int): Number of results to return for both. Defaults to 10.
        
        Returns:
            Dict[str, list[Dict[str, Any]]]: Combined dictionary with 'recent_games' and 'searched_games'.
        
        Raises:
            ValueError: If query is missing.
        """
        if not query:
            raise ValueError("Query is required for fetching both.")

        recent = self.get_recent_games(limit=limit)
        searched = self.search_games(query=query, limit=limit)
        return {"recent_games": recent, "searched_games": searched}

# Example usage (for testing)
if __name__ == "__main__":
    tool = IGDBTool()
    # Example: Fetch both recent and searched games
    results = tool.fetch_both(query="GTA", limit=5)
    print(results)