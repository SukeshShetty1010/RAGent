# ingest/ragent_ingestor.py — FINAL VERSION (November 09, 2025)
import logging
import json
import time
import os
import random
from typing import List
from datetime import datetime, UTC

from ingest.chunking import chunk_documents
from ingest.upsert import upsert_chunks
from vector.index_manager import create_index_if_not_exists
from data.news import NewsTool
from data.igdb import IGDBTool
from langchain_core.documents import Document
from utils.gpu_utils import get_device
from agent.constants import SOURCE_NEWS, SOURCE_IGDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _clean_content(raw: str) -> str:
    return raw.replace("[Upgrade subscription plan]", "") \
              .replace("Premium content", "") \
              .replace("Subscribe to read", "") \
              .strip()

def normalize_news(response: dict) -> List[Document]:
    docs = []
    seen_urls = set()
    
    for section in ("news", "headlines"):
        results = response.get(section, {}).get("results", [])
        logger.info(f"  {section.title()}: {len(results)} items")
        
        for item in results:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = item.get("title", "No title").strip()
            desc = item.get("description", "").strip()
            body = item.get("body", "") or item.get("content", "")
            full_text = f"{title}\n{desc}\n{body}".strip()
            
            if len(full_text) < 80:
                continue

            published_at = item.get("publishedAt") or datetime.now(UTC).isoformat()

            metadata = {
                "article_id": hash(url),
                "created_at": published_at,
                "source": SOURCE_NEWS,
                "title": title[:200],
                "url": url,
                "publisher": item.get("source", "Unknown")
            }
            docs.append(Document(page_content=full_text, metadata=metadata))
    
    logger.info(f"Normalized {len(docs)} fresh GNews articles")
    return docs

def normalize_igdb(response: dict) -> List[Document]:
    docs = []
    for key in ("recent_games", "searched_games"):
        games = response.get(key, [])
        for g in games:
            name = g.get("name", "Unknown Game")
            summary = g.get("summary", "")[:500]
            genres = ", ".join(gg["name"] for gg in g.get("genres", []))
            plats = ", ".join(p["name"] for p in g.get("platforms", []))
            content = f"{name}\n{summary}\nGenres: {genres}\nPlatforms: {plats}".strip()
            if len(content) < 50:
                continue
            ts = g.get("first_release_date")
            created_at = datetime.fromtimestamp(ts, tz=UTC).isoformat() if ts else datetime.now(UTC).isoformat()
            metadata = {
                "article_id": g.get("id"),
                "created_at": created_at,
                "source": SOURCE_IGDB,
                "title": name
            }
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

def log_metrics(metrics: dict):
    os.makedirs("eval", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%H-%M-%S")
    path = f"eval/run_{ts}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved: {path}")

def main():
    device = get_device()
    logger.info(f"Starting GPU ingestion on {device.upper()}")

    create_index_if_not_exists()

    try:
        news_tool = NewsTool()
        igdb_tool = IGDBTool()
        start_total = time.time()

        # PERMANENT SOLUTION: NO QUERY HACKS
        logger.info("Fetching FRESH gaming news (broad feed + query-aware)...")
        t0 = time.time()
        news_resp = news_tool.fetch_both(
            query="",           # EMPTY = FULL GAMING FEED (IGN, GameSpot, Eurogamer, etc.)
            limit=50            # 50 articles every run → KB stays HOT
        )
        news_docs = normalize_news(news_resp)
        news_chunks = chunk_documents(news_docs) if news_docs else []
        if news_chunks:
            upsert_chunks(news_chunks)
        news_time = time.time() - t0

        # IGDB: Keep it random but safe
        logger.info("Fetching TRENDING games from IGDB...")
        t0 = time.time()
        SAFE_QUERIES = [
            "2025",
            "open world",
            "RPG",
            "shooter",
            "nintendo",
            "playstation",
            "xbox",
            "indie"
        ]
        igdb_query = random.choice(SAFE_QUERIES)
        logger.info(f"IGDB query: {igdb_query}")
        igdb_resp = igdb_tool.fetch_both(query=igdb_query, limit=25)
        igdb_docs = normalize_igdb(igdb_resp)
        igdb_chunks = chunk_documents(igdb_docs) if igdb_docs else []
        if igdb_chunks:
            upsert_chunks(igdb_chunks)
        igdb_time = time.time() - t0

        total_time = time.time() - start_total
        metrics = {
            "run_timestamp": datetime.now(UTC).isoformat(),
            "news_docs": len(news_docs),
            "igdb_docs": len(igdb_docs),
            "total_latency": round(total_time, 3),
            "status": "SUCCESS — PERMANENT GAMING KB"
        }
        log_metrics(metrics)
        logger.info(f"INGESTION COMPLETE — {len(news_docs)} news + {len(igdb_docs)} games")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()