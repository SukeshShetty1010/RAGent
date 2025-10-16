from ingest.utils import BaseIngestor
from typing import List, Dict

class NewsIngestor(BaseIngestor):
    def fetch(self) -> List[Dict]:
        """
        Fetch latest gaming or tech news from APITube.io.
        """
        self.logger.info("Fetching news data from APITube.io...")
        # Example endpoint — replace with your actual API call
        response = self.api_client.get_news(category="gaming", limit=25)
        return response

    def normalize(self, articles: List[Dict]) -> List[Dict]:
        """
        Convert raw news data into a uniform schema.
        """
        self.logger.info("Normalizing news data...")
        normalized = []
        for a in articles:
            normalized.append({
                "title": a.get("title"),
                "description": a.get("description") or a.get("content"),
                "published_at": a.get("publishedAt"),
                "url": a.get("url"),
                "tags": a.get("keywords", []),
                "source": "APITube.io"
            })
        return normalized


if __name__ == "__main__":
    from your_api_clients.apitube_client import apitube_client  # replace with actual import path

    output_file = "data/raw/news_data.jsonl"
    ingestor = NewsIngestor(api_client=apitube_client, output_path=output_file)
    ingestor.run()