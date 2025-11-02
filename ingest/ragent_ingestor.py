# ingest/ragent_ingestor.py
import logging
import json
import time
import os
from typing import List
from datetime import datetime, UTC

from ingest.chunking import chunk_documents
from ingest.upsert import upsert_chunks
from vector.index_manager import create_index_if_not_exists
from data.news import NewsTool
from data.igdb import IGDBTool
from langchain_core.documents import Document
from utils.gpu_utils import get_device  # GPU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _clean_content(raw: str) -> str:
    return raw.replace("[Upgrade subscription plan]", "") \
              .replace("Premium content", "") \
              .replace("Subscribe to read", "") \
              .strip()

def normalize_news(response: dict) -> List[Document]:
    docs = []
    for section in ("news", "headlines"):
        results = response.get(section, {}).get("results", [])
        logger.info(f"  {section.title()}: {len(results)} items")
        for item in results:
            content = _clean_content(f"{item.get('title','')}\n{item.get('description','')}\n{item.get('body','')}")
            if not content:
                continue
            metadata = {
                "article_id": item.get("id"),
                "created_at": item.get("published_at", datetime.now(UTC).isoformat()),
                "source": "news"
            }
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

def normalize_igdb(response: dict) -> List[Document]:
    docs = []
    for key in ("recent_games", "searched_games"):
        games = response.get(key, [])
        for g in games:
            genres = ", ".join(gg["name"] for gg in g.get("genres", []))
            plats = ", ".join(p["name"] for p in g.get("platforms", []))
            content = f"{g.get('name','')}\n{g.get('summary','')}\nGenres: {genres}\nPlatforms: {plats}"
            ts = g.get("first_release_date")
            created_at = datetime.fromtimestamp(ts, tz=UTC).isoformat() if ts else datetime.now(UTC).isoformat()
            metadata = {
                "article_id": g.get("id"),
                "created_at": created_at,
                "source": "igdb"
            }
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

def log_metrics(metrics: dict):
    os.makedirs("eval", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"eval/run_{ts}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved: {path}")

if __name__ == "__main__":
    device = get_device()
    logger.info(f"Starting GPU ingestion on {device.upper()}")

    create_index_if_not_exists()

    try:
        news_tool = NewsTool()
        igdb_tool = IGDBTool()
        start_total = time.time()

        # === NEWS ===
        logger.info("Fetching news...")
        t0 = time.time()
        news_resp = news_tool.fetch_both(query="gaming", limit=15, country="us")
        news_docs = normalize_news(news_resp)
        news_chunks = chunk_documents(news_docs) if news_docs else []
        if news_chunks:
            upsert_chunks(news_chunks)
        news_time = time.time() - t0

        # === IGDB ===
        logger.info("Fetching IGDB games...")
        t0 = time.time()
        igdb_resp = igdb_tool.fetch_both(query="GTA OR shooter OR simulator", limit=10)
        igdb_docs = normalize_igdb(igdb_resp)
        igdb_chunks = chunk_documents(igdb_docs) if igdb_docs else []
        if igdb_chunks:
            upsert_chunks(igdb_chunks)
        igdb_time = time.time() - t0

        # === METRICS ===
        total_time = time.time() - start_total
        metrics = {
            "run_timestamp": datetime.now(UTC).isoformat(),
            "device": device,
            "news_latency": round(news_time, 3),
            "igdb_latency": round(igdb_time, 3),
            "total_latency": round(total_time, 3),
            "news_docs": len(news_docs),
            "news_chunks": len(news_chunks),
            "igdb_docs": len(igdb_docs),
            "igdb_chunks": len(igdb_chunks),
            "automation_depth": 5
        }
        log_metrics(metrics)
        logger.info(f"Ingestion complete in {total_time:.2f}s")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise