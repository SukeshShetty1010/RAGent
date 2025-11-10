# data/news.py — FINAL BULLETPROOF VERSION (Nov 10, 2025)
import os
import requests
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class NewsTool:
    def __init__(self):
        # DO NOT CRASH ON IMPORT — lazy check
        pass

    def _get_key(self):
        key = os.getenv("MEDIASTACK_KEY")
        if not key:
            raise ValueError(
                "MEDIASTACK_KEY not found! Add to your shell:\n"
                "export MEDIASTACK_KEY=your_actual_key_here\n"
                "Or run: echo 'export MEDIASTACK_KEY=your_key' >> ~/.bashrc"
            )
        return key

    def fetch_both(self, query: str = "", limit: int = 40, **filters) -> Dict[str, Any]:
        api_key = self._get_key()  # Only now we check + crash if missing
        
        params = {
            "access_key": api_key,
            "languages": "en",
            "limit": min(limit, 100),
            "categories": "technology,entertainment",
            "keywords": "gaming OR esports OR ubisoft OR rockstar OR nintendo OR playstation OR xbox OR ea OR activision OR valve"
        }
        if query:
            # Smart boost: put query first
            params["keywords"] = f"{query} " + params["keywords"]

        try:
            r = requests.get("http://api.mediastack.com/v1/news", params=params, timeout=12)
            r.raise_for_status()
            data = r.json().get("data", [])
            
            results = []
            for a in data[:limit]:
                desc = a.get("description") or ""
                results.append({
                    "title": a["title"],
                    "description": desc,
                    "body": (desc + " " + (a.get("content") or ""))[:1200],
                    "url": a["url"],
                    "publishedAt": a["published_at"],
                    "source": a.get("source", "MediaStack")
                })
            logger.info(f"MediaStack → {len(results)} fresh gaming articles")
            return {
                "news": {"results": results},
                "headlines": {"results": results[:limit//2]}
            }
        except Exception as e:
            logger.error(f"MediaStack failed: {e}")
            return {"news": {}, "headlines": {}}