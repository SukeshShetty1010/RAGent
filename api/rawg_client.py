import os
import requests
from dotenv import load_dotenv

load_dotenv()
RAWG_API_KEY = os.getenv("RAWG_API_KEY")
if not RAWG_API_KEY:
    raise RuntimeError("Missing RAWG_API_KEY in environment")

def search_games(query, page_size=10):
    """
    Search for games on RAWG API.
    
    Args:
        query (str): Search keyword or title.
        page_size (int): Number of results to return (default: 10).
    
    Returns:
        list: List of game dicts from search results.
    """
    endpoint = "https://api.rawg.io/api/games"
    params = {
        "key": RAWG_API_KEY,
        "search": query,
        "page_size": page_size
    }
    resp = requests.get(endpoint, params=params)
    if resp.status_code != 200:
        print(f"Error querying RAWG search: {resp.status_code} {resp.text}")
        return []
    data = resp.json()
    return data.get("results", [])

def get_game_details(game_id):
    """
    Fetch detailed metadata for a specific game by ID.
    
    Args:
        game_id (int or str): The RAWG game ID.
    
    Returns:
        dict: Full game details or None on error.
    """
    endpoint = f"https://api.rawg.io/api/games/{game_id}"
    params = {
        "key": RAWG_API_KEY
    }
    resp = requests.get(endpoint, params=params)
    if resp.status_code != 200:
        print(f"Error fetching RAWG details: {resp.status_code} {resp.text}")
        return None
    return resp.json()