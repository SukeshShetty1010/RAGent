# ingest/game_ingest.py

from ingest.utils import BaseIngestor, setup_logger
from typing import List, Dict
from vector.embeddings import generate_embeddings
from vector.weaviate_client import ensure_schema, add_data


class GameIngestor(BaseIngestor):
    """
    Ingestor for fetching, normalizing, and storing game data from IGDB.
    Integrates with Weaviate vector DB.
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
                "fields name, summary, first_release_date, genres.name, platforms.name;"
                "sort first_release_date desc; limit 20;"
            )
        )
        return response

    def normalize(self, games: List[Dict]) -> List[Dict]:
        """
        Convert raw IGDB data into a uniform schema.
        """
        self.logger.info("🧩 Normalizing game data...")
        normalized = []
        for g in games:
            normalized.append({
                "title": g.get("name"),
                "description": g.get("summary"),
                "release_date": g.get("first_release_date"),
                "genres": [genre["name"] for genre in g.get("genres", [])] if g.get("genres") else [],
                "platforms": [p["name"] for p in g.get("platforms", [])] if g.get("platforms") else [],
                "source": "IGDB"
            })
        return normalized

    def push_to_vector_db(self, games: List[Dict]):
        """
        Generate embeddings for each game and push to Weaviate.
        """
        self.logger.info("📡 Generating embeddings and pushing to Weaviate...")
        ensure_schema()
        pushed = 0

        for g in games:
            text = g.get("description", "")
            if not text:
                continue
            embedding = generate_embeddings([text])[0]
            try:
                add_data("Games", g, vector=embedding)
                pushed += 1
            except Exception as e:
                self.logger.error(f"Failed to push {g.get('title')}: {e}")

        self.logger.info(f"✅ Successfully pushed {pushed} games to Weaviate.")

    def run(self):
        """
        Complete end-to-end ingestion pipeline.
        1. Fetch data
        2. Normalize it
        3. Save locally
        4. Push to Weaviate
        """
        self.logger.info("🚀 Starting Game Ingestion Pipeline...")
        raw_games = self.fetch()
        normalized_games = self.normalize(raw_games)
        self.save(normalized_games)
        self.push_to_vector_db(normalized_games)
        self.logger.info("🏁 Game ingestion pipeline completed successfully!")


if __name__ == "__main__":
    from api.igdb_client import igdb_client  # ✅ Adjust to your actual client import path

    output_file = "data/raw/game_data.jsonl"
    ingestor = GameIngestor(api_client=igdb_client, output_path=output_file)
    ingestor.run()
