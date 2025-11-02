# ingest/news_ingest.py
from ingest.utils import BaseIngestor
from typing import List, Dict
from langchain_core.documents import Document
from datetime import datetime, UTC
from ingest.chunking import chunk_documents
from ingest.upsert import upsert_chunks
from vector.index_manager import create_index_if_not_exists

class NewsIngestor(BaseIngestor):
    def fetch(self) -> List[Dict]:
        self.logger.info("Fetching news from APITube.io...")
        response = self.api_client.get_news(category="gaming", limit=25)
        return response

    def normalize(self, articles: List[Dict]) -> List[Document]:
        self.logger.info("Normalizing news data...")
        docs = []
        for a in articles:
            title = a.get("title", "")
            description = a.get("description") or a.get("content") or ""
            content = f"{title}\n{description}".strip()
            if not content:
                continue

            published_at = a.get("publishedAt") or a.get("published_at")
            try:
                created_at = datetime.fromisoformat(published_at.replace("Z", "+00:00")).isoformat()
            except:
                created_at = datetime.now(UTC).isoformat()

            metadata = {
                "article_id": a.get("id") or hash(content),
                "created_at": created_at,
                "source": "APITube.io"
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def run(self):
        self.logger.info("Starting News Ingestion Pipeline...")
        create_index_if_not_exists()

        raw_articles = self.fetch()
        docs = self.normalize(raw_articles)

        if not docs:
            self.logger.warning("No news documents to ingest.")
            return

        chunks = chunk_documents(docs)
        upsert_chunks(chunks)  # GPU BATCH EMBEDDING

        normalized_dicts = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        super().save_jsonl(normalized_dicts, self.output_path)

        self.logger.info("News ingestion completed!")

if __name__ == "__main__":
    from api.apitube_client import apitube_client  # Update path
    output_file = "data/raw/news_data.jsonl"
    ingestor = NewsIngestor(api_client=apitube_client, output_path=output_file)
    ingestor.run()