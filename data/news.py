# data/news.py — THE FINAL VERSION (NO MORE QUERY HACKS)
import os
import requests
import logging
from typing import Dict, Any, List
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

class NewsTool:
    def __init__(self):
        self.api_key = os.getenv("GNEWS_API_KEY")
        if not self.api_key:
            raise ValueError("GNEWS_API_KEY missing! Get free key: https://gnews.io")
        self.base = "https://gnews.io/api/v4"

    def _get(self, endpoint: str, params: dict) -> List[Dict]:
        url = f"{self.base}/{endpoint}"
        params.update({
            "apikey": self.api_key,
            "lang": "en",
            "country": "us",
            "max": 100
        })
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            return data.get("articles", [])
        except Exception as e:
            logger.error(f"GNews {endpoint} failed: {e}")
            return []

    def fetch_both(self, query: str = "", limit: int = 40, **filters) -> Dict[str, Any]:
        # CLEAN QUERY — remove dangerous chars
        clean_q = ""
        if query:
            clean_q = "".join(c for c in query if c.isalnum() or c in " -_").strip()
            clean_q = clean_q[:100]  # GNews limit

        # ALWAYS fetch fresh gaming news — even for "what is 2+2?"
        search_articles = []
        if clean_q:
            search_articles = self._get("search", {"q": clean_q})

        # THIS IS THE MAGIC: BROAD GAMING FEED — ALWAYS FRESH
        broad_articles = self._get("top-headlines", {
            "category": "technology",
            "q": "gaming OR esports OR nintendo OR playstation OR xbox OR ubisoft OR rockstar OR ea OR activision OR valve OR epic OR indie"
        })

        # Combine + dedupe by URL
        seen = set()
        unique = []
        for art in search_articles + broad_articles:
            url = art["url"]
            if url not in seen:
                seen.add(url)
                unique.append(art)

        # Sort newest first
        unique.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)

        # Format
        results = []
        for a in unique[:limit]:
            results.append({
                "title": a["title"],
                "description": a.get("description", "") or "",
                "body": a.get("content", "")[:1200],
                "url": a["url"],
                "publishedAt": a["publishedAt"],
                "source": a["source"]["name"]
            })

        logger.info(f"GNews fetched {len(results)} articles (query='{query}')")
        return {
            "news": {"results": results},
            "headlines": {"results": results[:limit//2]}
        }