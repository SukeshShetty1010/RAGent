# tools/news.py
from typing import Dict, Any, Optional
from api.apitube_client import APITubeClient  # Assuming apitube_client.py is in the same directory or importable

class NewsTool:
    """
    A tool for fetching news articles and top headlines using the APITube API.
    This tool can perform general news searches, retrieve top headlines, or both, with optional filters.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the NewsTool with an optional API key.
        Falls back to environment variable if not provided.
        """
        self.client = APITubeClient(api_key)

    def run(self, query: str | None = None, top_headlines: bool = False, limit: int = 10, **filters) -> Dict[str, Any]:
        """
        Run the news fetch operation.
        
        Args:
            query (str, optional): The search query for general news. Required if not fetching top headlines.
            top_headlines (bool): If True, fetch top headlines instead of searching. Defaults to False.
            limit (int): Number of results to return (for general search). Defaults to 10.
            **filters: Additional filters as keyword arguments (e.g., country='us', category='technology').
        
        Returns:
            Dict[str, Any]: JSON response from the API containing news data.
        
        Raises:
            ValueError: If query is missing when not fetching top headlines.
        """
        if top_headlines:
            return self.client.get_top_headlines(**filters)
        else:
            if not query:
                raise ValueError("Query is required for general news search.")
            return self.client.get_news(q=query, limit=limit, **filters)

    def fetch_both(self, query: str, limit: int = 10, **filters) -> Dict[str, Any]:
        """
        Fetch both general news and top headlines without storing to a file.
        
        Args:
            query (str): The search query for general news.
            limit (int): Number of results to return for general news. Defaults to 10.
            **filters: Additional filters applied to both requests (e.g., country='us', category='technology').
        
        Returns:
            Dict[str, Any]: Combined dictionary with 'news' and 'headlines'.
        
        Raises:
            ValueError: If query is missing.
        """
        if not query:
            raise ValueError("Query is required for fetching both.")

        news = self.run(query=query, limit=limit, **filters)
        headlines = self.run(top_headlines=True, **filters)
        return {"news": news, "headlines": headlines}

# Example usage (for testing)
if __name__ == "__main__":
    tool = NewsTool()
    # Example: Fetch both without storing
    results = tool.fetch_both(query="GTA", limit=5, country="us", category="technology")
    print(results)