# ingest/game_ingest.py
from ingest.utils import BaseIngestor, setup_logger
from typing import List, Dict
from langchain_core.documents import Document
from datetime import datetime, UTC
from ingest.chunking import chunk_documents
from ingest.upsert import upsert_chunks
from vector.index_manager import create_index_if_not_exists
from agent.constants import SOURCE_IGDB 

class GameIngestor(BaseIngestor):
    def fetch(self) -> List[Dict]:
        self.logger.info("Fetching game data from IGDB...")
        response = self.api_client.api_request(
            endpoint="games",
            data=(
                "fields id, name, summary, first_release_date, genres.name, platforms.name;"
                "sort first_release_date desc; limit 20;"
            )
        )
        return response

    def normalize(self, games: List[Dict]) -> List[Document]:
        self.logger.info("Normalizing game data...")
        docs = []
        for g in games:
            genres = ', '.join([genre["name"] for genre in g.get("genres", [])])
            platforms = ', '.join([p["name"] for p in g.get("platforms", [])])
            content = f"{g.get('name', '')}\n{g.get('summary', '')}\nGenres: {genres}\nPlatforms: {platforms}"
            
            release_timestamp = g.get("first_release_date")
            created_at = (
                datetime.fromtimestamp(release_timestamp, tz=UTC).isoformat() 
                if release_timestamp else 
                datetime.now(UTC).isoformat()
            )
            metadata = {
                "article_id": g.get("id"),
                "created_at": created_at,
                "source": SOURCE_IGDB  # ← FIXED: was "IGDB"
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def run(self):
        self.logger.info("Starting Game Ingestion Pipeline...")
        create_index_if_not_exists()
        
        raw_games = self.fetch()
        docs = self.normalize(raw_games)
        
        if not docs:
            self.logger.warning("No documents normalized.")
            return
        
        chunks = chunk_documents(docs)
        upsert_chunks(chunks)  # GPU BATCH EMBEDDING HERE
        
        normalized_dicts = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        super().save_jsonl(normalized_dicts, self.output_path)
        
        self.logger.info("Game ingestion completed!")