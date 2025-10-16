from ingest.utils import BaseIngestor
from ingest.utils import setup_logger
from typing import List, Dict

class GameIngestor(BaseIngestor):
    def fetch(self) -> List[Dict]:
        """
        Fetch raw data from IGDB API.
        Assumes `self.api_client` is an initialized IGDB API client.
        """
        self.logger.info("Fetching game data from IGDB...")
        # Example IGDB API call – modify per your client
        response = self.api_client.api_request(
            endpoint="games",
            data="fields name, summary, first_release_date, genres.name, platforms.name;"
                  "sort first_release_date desc; limit 20;"
        )
        return response

    def normalize(self, games: List[Dict]) -> List[Dict]:
        """
        Convert raw IGDB data into a uniform schema.
        """
        self.logger.info("Normalizing game data...")
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


if __name__ == "__main__":
    from your_api_clients.igdb_client import igdb_client  # replace with actual import path

    output_file = "data/raw/game_data.jsonl"
    ingestor = GameIngestor(api_client=igdb_client, output_path=output_file)
    ingestor.run()