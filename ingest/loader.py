# ingest/loader.py
import json
from langchain_core.documents import Document
from typing import List
import logging
from datetime import datetime, UTC

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(json_path: str) -> List[Document]:
    """
    Loads news and headlines from the JSON file into LangChain Documents.
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
        sections = ['news', 'headlines']
        for section in sections:
            if section in data and 'results' in data[section]:
                for item in data[section]['results']:
                    # Combine specified fields into content as per user request
                    content = f"{item.get('title', '')}\n{item.get('description', '')}\n{item.get('body', '')}"
                    # Metadata fields as per user request
                    metadata = {
                        "article_id": item.get('id'),
                        "created_at": item.get('published_at', datetime.now(UTC).isoformat()),  # Fallback to current time
                        "source": item['source'].get('domain', '') if 'source' in item else ''
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
    json_path = r"D:\Sukesh\Cursor\RAG Project\news_and_headlines_2025-10-25_16-58-58.json"
    try:
        documents = load_documents(json_path)
        for doc in documents:
            logger.info(f"Document loaded: {doc.metadata}")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")