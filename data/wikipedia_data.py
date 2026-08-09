"""
data/wikipedia_data.py

Reusable utility to fetch Wikipedia article plaintext for RAG ingestion.
No API key required — MediaWiki's public action API.

Usage:
    from data.wikipedia_data import fetch_wikipedia_article
    article = fetch_wikipedia_article("Elden Ring")
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "RAGent/1.0 (gaming intelligence research bot)"
TIMEOUT = 15


def _get(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(
            API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _fetch_by_title(title: str) -> Optional[Dict[str, Any]]:
    data = _get(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "format": "json",
            "redirects": 1,
            "titles": title,
        }
    )
    if not data:
        return None

    pages = (data.get("query") or {}).get("pages") or {}
    for page_id, page in pages.items():
        if page_id == "-1" or "missing" in page:
            return None
        extract = page.get("extract")
        if not extract:
            return None
        return {
            "pageid": page.get("pageid"),
            "title": page.get("title"),
            "extract": extract,
        }
    return None


def _search_title(query: str) -> Optional[str]:
    data = _get(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        }
    )
    if not data:
        return None

    results = (data.get("query") or {}).get("search") or []
    if not results:
        return None
    return results[0].get("title")


def fetch_wikipedia_article(title: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a Wikipedia article's plaintext extract by title.

    Tries an exact `titles=` lookup (with redirect following) first. If
    the page doesn't exist under that exact title, retries once via
    `list=search`. Returns None if no matching page is found — no
    guessing, since the honesty gate depends on this being accurate.
    """
    if not title or not title.strip():
        return None

    page = _fetch_by_title(title)
    if page is not None:
        return page

    searched_title = _search_title(title)
    if not searched_title or searched_title == title:
        return None
    return _fetch_by_title(searched_title)
