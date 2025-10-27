# ingest/game_ingest.py
from ingest.utils import BaseIngestor, setup_logger
from typing import List, Dict
from langchain_core.documents import Document
from datetime import datetime, UTC
from ingest.chunking import chunk_documents
from ingest.upsert import upsert_chunks
from vector.index_manager import create_index_if_not_exists

class GameIngestor(BaseIngestor):
    """
    Ingestor for fetching, normalizing, and storing game data from IGDB.
    Integrates with Weaviate vector DB using the shared pipeline.
    """

    def fetch(self) -> List[Dict]:
        """
        Fetch raw data from IGDB API.
        Assumes `self.api_client` is an initialized IGDB API client.
        """
        self.logger.info("🎮 Fetching game data from IGDB...")
        response = self.api_client.api_request(
            endpoint="games",
            data=(
                "fields id, name, summary, first_release_date, genres.name, platforms.name;"
                "sort first_release_date desc; limit 20;"
            )
        )
        return response

    def normalize(self, games: List[Dict]) -> List[Document]:
        """
        Convert raw IGDB data into LangChain Documents, matching the shared schema.
        """
        self.logger.info("🧩 Normalizing game data...")
        docs = []
        for g in games:
            # Combine fields into content (similar to news: title + description + body)
            genres = ', '.join([genre["name"] for genre in g.get("genres", [])])
            platforms = ', '.join([p["name"] for p in g.get("platforms", [])])
            content = f"{g.get('name', '')}\n{g.get('summary', '')}\nGenres: {genres}\nPlatforms: {platforms}"
            
            # Metadata aligned with shared schema
            release_timestamp = g.get("first_release_date")
            created_at = datetime.fromtimestamp(release_timestamp, tz=UTC).isoformat() if release_timestamp else datetime.now(UTC).isoformat()
            metadata = {
                "article_id": g.get("id"),
                "created_at": created_at,
                "source": "IGDB"
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def run(self):
        """
        Complete end-to-end ingestion pipeline:
        1. Ensure Weaviate index exists
        2. Fetch data
        3. Normalize to Documents
        4. Chunk documents
        5. Upsert to Weaviate (with duplicate checks and embeddings)
        """
        self.logger.info("🚀 Starting Game Ingestion Pipeline...")
        
        # Ensure the shared KnowledgeBase collection exists
        create_index_if_not_exists()
        
        raw_games = self.fetch()
        docs = self.normalize(raw_games)
        
        if not docs:
            self.logger.warning("No documents normalized from fetched data.")
            return
        
        # Chunk and upsert using the shared pipeline
        chunks = chunk_documents(docs)
        upsert_chunks(chunks)
        
        # Optionally save raw normalized data to JSONL (as in BaseIngestor)
        normalized_dicts = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        super().save_jsonl(normalized_dicts, self.output_path)
        
        self.logger.info("🏁 Game ingestion pipeline completed successfully!")

if __name__ == "__main__":
    from api.igdb_client import igdb_client  # Adjust to your actual IGDB client import path
    
    output_file = "data/raw/game_data.jsonl"
    ingestor = GameIngestor(api_client=igdb_client, output_path=output_file)
    ingestor.run()