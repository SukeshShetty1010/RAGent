import os
import requests
from dotenv import load_dotenv

load_dotenv()

APITUBE_API_KEY = os.getenv("APITUBE_API_KEY")
BASE_URL = "https://api.apitube.io/v1"

class APITubeClient:
    def __init__(self, api_key: str | None = None):
        # Prefer explicit key, fallback to env var
        key = api_key if api_key is not None else os.getenv("APITUBE_API_KEY")
        if not key:
            raise RuntimeError("APITube API key missing")
        self.api_key = key

    def _headers(self):
        return {"X-API-Key": self.api_key}

    def get_news(self, q: str = None, limit: int = 10, **filters):
        url = f"{BASE_URL}/news/everything"
        params = {"per_page": limit}
        if q:
            params["title"] = q
        params.update(filters)
        resp = requests.get(url, headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    def get_top_headlines(self, **filters):
        url = f"{BASE_URL}/news/top-headlines"
        resp = requests.get(url, headers=self._headers(), params=filters)
        resp.raise_for_status()
        return resp.json()

if __name__ == "__main__":
    client = APITubeClient()
    print(client.get_news(q="test", limit=2))
