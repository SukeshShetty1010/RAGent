# ingest/igdb_loader.py
import json
from langchain_core.documents import Document
from typing import List
import logging
from datetime import datetime, UTC

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_igdb_documents(json_path: str) -> List[Document]:
    """
    Loads game data from the IGDB JSON file into LangChain Documents.
    Logs the loading process for traceability as per RAGent requirements.
    
    Args:
        json_path (str): Path to the JSON file.
    
    Returns:
        List[Document]: List of documents.
    
    Raises:
        FileNotFoundError: If the JSON file is not found.
        json.JSONDecodeError: If the JSON file is invalid.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        docs = []
        if 'games' in data:
            for item in data['games']:
                # Combine fields into content (similar to news: title + description + body)
                genres = ', '.join([g['name'] for g in item.get('genres', [])])
                platforms = ', '.join([p['name'] for p in item.get('platforms', [])])
                content = f"{item.get('name', '')}\n{item.get('summary', '')}\nGenres: {genres}\nPlatforms: {platforms}"
                
                # Metadata fields aligned with news schema
                release_timestamp = item.get('first_release_date')
                created_at = datetime.fromtimestamp(release_timestamp, tz=UTC).isoformat() if release_timestamp else datetime.now(UTC).isoformat()
                metadata = {
                    "article_id": item.get('id'),
                    "created_at": created_at,
                    "source": "IGDB"
                }
                docs.append(Document(page_content=content, metadata=metadata))
        
        logger.info(f"Loaded {len(docs)} documents from {json_path}")
        return docs
    
    except FileNotFoundError:
        logger.error(f"JSON file not found at {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading documents: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the loader
    json_path = r"D:\Sukesh\Cursor\RAG Project\igdb_games_2025-10-26_15-07-24.json"
    try:
        documents = load_igdb_documents(json_path)
        for doc in documents:
            logger.info(f"Document loaded: {doc.metadata}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")