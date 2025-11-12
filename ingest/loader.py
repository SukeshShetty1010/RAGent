# ingest/loader.py
import json
import os
import logging
from typing import List, Dict, Any, Union
from datetime import datetime, timezone
from langchain_core.documents import Document

# optional imports for language detection and translation
try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# small slugify utility (no extra dependency)
def slugify(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(c if (c.isalnum() or c in "- ") else "-" for c in s)
    s = "-".join(s.split())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")

def _iso_now():
    return datetime.now(timezone.utc).isoformat()

def _normalize_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        # try comma separated
        if "," in value:
            return [v.strip() for v in value.split(",") if v.strip()]
        return [value.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if v is not None]
    return [str(value)]

def _detect_language(text: str) -> str:
    if not text or detect is None:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"

def _translate_to_english(text: str, src_lang: str) -> str:
    """
    Best-effort translation to English. Uses transformers Marian models if available.
    If translation fails or transformers not available, returns original text.
    """
    if not text:
        return text
    if src_lang in ("en", "unknown"):
        return text

    if not _TRANSFORMERS_AVAILABLE:
        logger.warning("transformers not available - skipping translation")
        return text

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer)
        out = translator(text, max_length=1024)
        if isinstance(out, list) and len(out) > 0:
            return out[0].get("translation_text", text)
        return text
    except Exception as e:
        logger.warning("Translation model unavailable for %s → en: %s", src_lang, e)
        return text

# Loader implementations for each source (keeps fields separate as requested)
def _make_slug(title: str, release_date: Union[str, None]) -> str:
    year = None
    if release_date:
        try:
            year = release_date.split("-")[0]
        except Exception:
            year = None
    base = title or "untitled"
    if year:
        return f"{slugify(base)}-{year}"
    return slugify(base)

def _doc_content_length(text: str) -> int:
    if not text:
        return 0
    # simple token estimate: whitespace-split
    return len(text.split())

def _wrap_document(content: str, metadata: Dict[str, Any]) -> Document:
    # language detection + best-effort translation to English (if needed)
    lang = _detect_language(content) if detect else "unknown"
    metadata["language"] = lang
    # attempt translation only if not english
    if lang != "en" and lang != "unknown":
        content_translated = _translate_to_english(content, lang)
        metadata["translated_from"] = lang
        # prefer translated text for embeddings & content length
        content = content_translated

    metadata["content_length"] = _doc_content_length(content)
    return Document(page_content=content, metadata=metadata)

# RAWG loader
def _load_rawg_records(data: Dict[str, Any]) -> List[Document]:
    docs = []
    if isinstance(data, dict) and "name" in data:
        data = [data]
    for g in data:
        title = g.get("name") or g.get("title")
        release_date = g.get("released") or g.get("release_date")
        slug = _make_slug(title, release_date)
        metadata = {
            "source": "rawg",
            "game_id": str(g.get("id")) if g.get("id") is not None else None,
            "unified_game_id": slug,
            "slug": slug,
            "title": title,
            "description": g.get("description"),
            "release_date": release_date,
            "release_year": int(release_date.split("-")[0]) if release_date and "-" in release_date else None,
            "created_at": _iso_now(),
            "genres": _normalize_list(g.get("genres")),
            "platforms": _normalize_list(g.get("platforms")),
            "developers": _normalize_list(g.get("developers")),
            "publishers": _normalize_list(g.get("publishers")),
            "tags": _normalize_list(g.get("tags")),
            "rating": g.get("rating"),
            "rating_count": g.get("ratings_count") or g.get("ratings"),
            "metacritic": g.get("metacritic"),
            "esrb_rating": g.get("esrb_rating"),
            "playtime": g.get("playtime"),
            "site_detail_url": g.get("website"),
        }
        content = f"Name: {title}\n\nDescription:\n{g.get('description','')}"
        docs.append(_wrap_document(content, metadata))
    return docs

# IGDB loader
def _load_igdb_records(data: Dict[str, Any]) -> List[Document]:
    docs = []
    if isinstance(data, dict) and "name" in data:
        data = [data]
    for g in data:
        title = g.get("name")
        release_date = g.get("first_release_date") or g.get("release_date")
        # igdb often has unix timestamps; try to parse
        rd_str = None
        try:
            if isinstance(release_date, int):
                rd_str = datetime.utcfromtimestamp(release_date).strftime("%Y-%m-%d")
            elif isinstance(release_date, str):
                rd_str = release_date
        except Exception:
            rd_str = None

        slug = _make_slug(title, rd_str)
        metadata = {
            "source": "igdb",
            "game_id": str(g.get("id")) if g.get("id") is not None else None,
            "unified_game_id": slug,
            "slug": slug,
            "title": title,
            "description": g.get("description") or g.get("summary"),
            "release_date": rd_str,
            "release_year": int(rd_str.split("-")[0]) if rd_str and "-" in rd_str else None,
            "created_at": _iso_now(),
            "genres": _normalize_list(g.get("genres")),
            "platforms": _normalize_list(g.get("platforms")),
            "developers": _normalize_list(g.get("involved_companies")),
            "publishers": _normalize_list(g.get("publishers")),
            "themes": _normalize_list(g.get("themes")),
            "franchise": ", ".join(_normalize_list(g.get("franchises"))),
            "rating": g.get("rating"),
        }
        content = f"Name: {title}\n\nDescription:\n{metadata.get('description','')}"
        docs.append(_wrap_document(content, metadata))
    return docs

# GameSpot loader
def _load_gamespot_records(data: Dict[str, Any]) -> List[Document]:
    docs = []
    if not data:
        return docs

    # If it's raw game object
    if isinstance(data, dict) and "Game Information" in data:
        game_info = data.get("Game Information", {})
        releases = data.get("Releases", [])
        articles = data.get("Articles", [])
        reviews = data.get("Reviews", [])

        title = game_info.get("name")
        release_date = game_info.get("release_date") or game_info.get("released")
        slug = _make_slug(title, release_date)

        # main game doc
        metadata = {
            "source": "gamespot",
            "game_id": str(game_info.get("id")) if game_info.get("id") is not None else None,
            "unified_game_id": slug,
            "slug": slug,
            "title": title,
            "description": game_info.get("description"),
            "release_date": release_date,
            "release_year": int(release_date.split("-")[0]) if release_date and "-" in release_date else None,
            "created_at": _iso_now(),
            "genres": _normalize_list(game_info.get("genres")),
            "themes": _normalize_list(game_info.get("themes")),
            "developers": _normalize_list(game_info.get("developers")),
            "publishers": _normalize_list(game_info.get("publishers")),
            "platforms": _normalize_list(game_info.get("platforms")),
            "franchise": game_info.get("franchise"),
            "site_detail_url": game_info.get("site_detail_url"),
            "articles_count": len(articles),
            "reviews_count": len(reviews),
        }
        content = f"Title: {title}\n\nDescription:\n{game_info.get('description','')}"
        docs.append(_wrap_document(content, metadata))

        # reviews and articles as separate docs
        for r in reviews:
            text = f"Review: {r.get('title','')}\n\n{r.get('deck','')}\n\nGood: {r.get('good')}\nBad: {r.get('bad')}"
            meta = {
                "source": "gamespot",
                "game_id": str(game_info.get("id")),
                "unified_game_id": slug,
                "slug": slug,
                "title": title,
                "description": r.get("deck"),
                "created_at": _iso_now(),
                "reviews_count": len(reviews),
                "platforms": _normalize_list(r.get("platforms")),
                "site_detail_url": r.get("site_detail_url"),
            }
            docs.append(_wrap_document(text, meta))

        for a in articles:
            text = f"Article: {a.get('title','')}\n\n{a.get('deck','')}"
            meta = {
                "source": "gamespot",
                "game_id": str(game_info.get("id")),
                "unified_game_id": slug,
                "slug": slug,
                "title": title,
                "description": a.get("deck"),
                "created_at": _iso_now(),
                "articles_count": len(articles),
                "site_detail_url": a.get("site_detail_url"),
            }
            docs.append(_wrap_document(text, meta))

    # If list-of-games form
    elif isinstance(data, list):
        for g in data:
            docs.extend(_load_gamespot_records({"Game Information": g, "Releases": [], "Articles": [], "Reviews": []}))
    else:
        logger.warning("Unrecognized GameSpot data format for loader.")

    return docs

def load_documents(source: Union[str, Dict[str, Any]]) -> List[Document]:
    """
    Load JSON (RAWG / IGDB / GameSpot) and return list of Documents.
    Detect source and dispatch to source-specific loader.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    docs = []
    try:
        if isinstance(data, dict):
            if "Game Information" in data or ("Games" in data and isinstance(data.get("Games"), list)):
                docs = _load_gamespot_records(data)
            elif "name" in data and ("metacritic" in data or "esrb_rating" in data):
                docs = _load_rawg_records(data)
            elif "name" in data:
                docs = _load_igdb_records(data)
            elif isinstance(data.get("results"), list):
                # RAWG search results
                docs = _load_rawg_records(data.get("results"))
            else:
                # fallback: try all loaders
                docs = _load_gamespot_records(data) or _load_rawg_records(data) or _load_igdb_records(data)
        elif isinstance(data, list):
            # Try to detect by content of the first item
            first = data[0] if data else {}
            if "metacritic" in first or "esrb_rating" in first:
                docs = _load_rawg_records(data)
            elif "name" in first and "summary" in first:
                docs = _load_igdb_records(data)
            else:
                docs = _load_gamespot_records({"Game Information": first, "Releases": [], "Articles": [], "Reviews": []})
        else:
            logger.warning("Unrecognized data format.")
    except Exception as e:
        logger.exception("Failed to load documents: %s", e)
        return []

    logger.info("Loaded %d documents from detected source.", len(docs))
    return docs
