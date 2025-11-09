# RAG_ENT/agent/constants.py
"""
Single source of truth for the entire project.
Change here ONCE → works everywhere.
"""

# ==== SOURCES ====
SOURCE_NEWS = "news"
SOURCE_IGDB = "igdb"

SOURCE_DISPLAY = {
    SOURCE_NEWS: "News",
    SOURCE_IGDB: "IGDB Games"
}

# ==== WEAVIATE ====
WEAVIATE_INDEX = "KnowledgeBase"
TEXT_KEY = "text"

# ==== SEARCH DEFAULTS ====
DEFAULT_TOP_K = 5
NEWS_SCORE_THRESHOLD = 0.2
IGDB_SCORE_THRESHOLD = 0.7
HYBRID_ALPHA = 0.9  # 0.9 = heavy semantic, 0.1 = heavy keyword

# ==== CITATION KEYS ====
CITATION_FIELDS = ["article_id", "source", "created_at", "url", "title"]