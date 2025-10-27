# ragent_ingestor.py
import logging
import json
import time
import os
from ingest.loader import load_documents
from ingest.igdb_loader import load_igdb_documents
from ingest.chunking import chunk_documents
from ingest.upsert import upsert_chunks
from vector.index_manager import create_index_if_not_exists
from data.news import NewsTool
from data.igdb import IGDBTool
from langchain_core.documents import Document
from datetime import datetime, UTC

logging.basicConfig(level=logging.INFO)

def normalize_api_response(response: dict, source: str) -> list[Document]:
    """
    Normalize API response into LangChain Documents with consistent metadata.
    """
    docs = []
    if source == "news":
        sections = ['news', 'headlines']
        for section in sections:
            if section in response and 'results' in response[section]:
                for item in response[section]['results']:
                    content = f"{item.get('title', '')}\n{item.get('description', '')}\n{item.get('body', '')}"
                    metadata = {
                        "article_id": item.get('id'),
                        "created_at": item.get('published_at', datetime.now(UTC).isoformat()),
                        "source": item['source'].get('domain', '') if 'source' in item else source
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
    elif source == "igdb":
        if 'recent_games' in response or 'searched_games' in response:
            for games in [response.get('recent_games', []), response.get('searched_games', [])]:
                for item in games:
                    genres = ', '.join([g['name'] for g in item.get('genres', [])])
                    platforms = ', '.join([p['name'] for p in item.get('platforms', [])])
                    content = f"{item.get('name', '')}\n{item.get('summary', '')}\nGenres: {genres}\nPlatforms: {platforms}"
                    release_timestamp = item.get('first_release_date')
                    created_at = datetime.fromtimestamp(release_timestamp, tz=UTC).isoformat() if release_timestamp else datetime.now(UTC).isoformat()
                    metadata = {
                        "article_id": item.get('id'),
                        "created_at": created_at,
                        "source": source
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
    return docs

if __name__ == "__main__":
    # Create Weaviate index if not exists
    create_index_if_not_exists()
    
    try:
        # Initialize API tools and start timing
        news_tool = NewsTool()
        igdb_tool = IGDBTool()
        start_time = time.time()

        # Fetch fresh news data
        news_start = time.time()
        news_response = news_tool.fetch_both(query="technology", limit=5, country="us")
        news_docs = normalize_api_response(news_response, source="news")
        news_chunks = chunk_documents(news_docs) if news_docs else []
        upsert_chunks(news_chunks)
        news_latency = time.time() - news_start

        # Fetch fresh IGDB data
        igdb_start = time.time()
        igdb_response = igdb_tool.fetch_both(query="GTA", limit=5)
        igdb_docs = normalize_api_response(igdb_response, source="igdb")
        igdb_chunks = chunk_documents(igdb_docs) if igdb_docs else []
        upsert_chunks(igdb_chunks)
        igdb_latency = time.time() - igdb_start

        # Total latency
        total_latency = time.time() - start_time

        # Log metrics
        metrics = {
            "run_timestamp": datetime.now(UTC).isoformat(),
            "news_latency": news_latency,
            "igdb_latency": igdb_latency,
            "total_latency": total_latency,
            "news_chunks_upserted": len(news_chunks),
            "igdb_chunks_upserted": len(igdb_chunks),
            "automation_depth": 4  # fetch, normalize, chunk, upsert
        }
        os.makedirs("eval", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"eval/run_{timestamp}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics logged to eval/run_{timestamp}.json")

    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")