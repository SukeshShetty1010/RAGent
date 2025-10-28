# ragent_ingestor.py
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
        logger.info(f"  {section.capitalize()} results: {len(results)}")
        for item in results:
            title = item.get("title", "")
            desc  = item.get("description", "")
            body  = item.get("body", "")
            raw   = f"{title}\n{desc}\n{body}"
            content = _clean_content(raw)
            if not content:
                continue

            # FIXED: Ensure source="news"
            metadata = {
                "article_id": item.get("id"),
                "created_at": item.get("published_at", datetime.now(UTC).isoformat()),
                "source": "news"  # ← CRITICAL
            }
            docs.append(Document(page_content=content, metadata=metadata))
    return docs


def normalize_igdb(response: dict) -> List[Document]:
    docs = []
    for key in ("recent_games", "searched_games"):
        games = response.get(key, [])
        for g in games:
            genres = ", ".join([gg["name"] for gg in g.get("genres", [])])
            plats  = ", ".join([p["name"] for p in g.get("platforms", [])])
            content = f"{g.get('name','')}\n{g.get('summary','')}\nGenres: {genres}\nPlatforms: {plats}"
            ts = g.get("first_release_date")
            created_at = (datetime.fromtimestamp(ts, tz=UTC).isoformat()
                          if ts else datetime.now(UTC).isoformat())
            metadata = {
                "article_id": g.get("id"),
                "created_at": created_at,
                "source": "igdb"  # ← Also ensure
            }
            docs.append(Document(page_content=content, metadata=metadata))
    return docs


def log_metrics(metrics: dict):
    os.makedirs("eval", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"eval/run_{ts}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics → {path}")


if __name__ == "__main__":
    create_index_if_not_exists()

    try:
        news_tool = NewsTool()
        igdb_tool = IGDBTool()
        overall_start = time.time()

        # 1. NEWS
        logger.info("Fetching NEWS (gaming)…")
        news_start = time.time()
        news_resp = news_tool.fetch_both(query="gaming", limit=15, country="us")
        logger.info(f"Raw news keys: {list(news_resp.keys())}")
        news_docs = normalize_news(news_resp)
        logger.info(f"Normalized NEWS docs: {len(news_docs)}")
        news_chunks = chunk_documents(news_docs) if news_docs else []
        if news_chunks:
            upsert_chunks(news_chunks)
            logger.info(f"Upserted {len(news_chunks)} NEWS chunks")  # ← CLEAR LOG
        news_latency = time.time() - news_start

        # 2. IGDB
        logger.info("Fetching IGDB…")
        igdb_start = time.time()
        igdb_resp = igdb_tool.fetch_both(query="GTA OR shooter OR simulator", limit=10)
        logger.info(f"Raw IGDB keys: {list(igdb_resp.keys())}")
        igdb_docs = normalize_igdb(igdb_resp)
        logger.info(f"Normalized IGDB docs: {len(igdb_docs)}")
        igdb_chunks = chunk_documents(igdb_docs) if igdb_docs else []
        if igdb_chunks:
            upsert_chunks(igdb_chunks)
            logger.info(f"Upserted {len(igdb_chunks)} IGDB chunks")  # ← CLEAR LOG
        igdb_latency = time.time() - igdb_start

        # 3. METRICS
        total_latency = time.time() - overall_start
        metrics = {
            "run_timestamp": datetime.now(UTC).isoformat(),
            "news_latency": round(news_latency, 3),
            "igdb_latency": round(igdb_latency, 3),
            "total_latency": round(total_latency, 3),
            "news_chunks_upserted": len(news_chunks),
            "igdb_chunks_upserted": len(igdb_chunks),
            "automation_depth": 4
        }
        log_metrics(metrics)

    except Exception as e:
        logger.error(f"Ingestor crashed: {e}", exc_info=True)
        raise