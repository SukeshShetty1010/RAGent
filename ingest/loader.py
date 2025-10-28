# ingest/loader.py
import json
from langchain_core.documents import Document
from typing import List
import logging
from datetime import datetime, UTC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(json_path: str) -> List[Document]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        docs = []
        sections = ['news', 'headlines']
        for section in sections:
            if section in data and 'results' in data[section]:
                for item in data[section]['results']:
                    raw = f"{item.get('title','')}\n{item.get('description','')}\n{item.get('body','')}"
                    # Clean paywall junk
                    content = raw.replace("[Upgrade subscription plan]", "").replace("Premium content", "").strip()
                    if not content:
                        continue

                    metadata = {
                        "article_id": item.get('id'),
                        "created_at": item.get('published_at', datetime.now(UTC).isoformat()),
                        "source": item['source'].get('domain', 'unknown') if 'source' in item else 'unknown'
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
        
        logger.info(f"Loaded {len(docs)} clean documents from {json_path}")
        return docs
    
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise