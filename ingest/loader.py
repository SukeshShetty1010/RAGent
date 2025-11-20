# ingest/loader.py
"""
Patched loader for RAG_ent ingestion pipeline.

Produces:
 - canonical objects (plain dicts) with raw_source_blob stored as JSON string
 - content docs (plain dicts) suitable for chunking/upsert: {"text": ..., "metadata": {...}}

Normalization & policies applied:
 - game_id -> int or None
 - release_year -> int or None
 - arrays normalized to List[str] via normalize_simple_list
 - platforms canonicalized via canonicalize_platforms
 - language defaults to "en"
 - GameSpot reviews/articles filtered to items referencing the same game (by game id or slug in URL)
 - all outputs JSON-serializable (no custom objects)
"""

import json
import os
import logging
from typing import Any, Dict, List, Tuple, Optional, Union
from datetime import datetime, timezone

from data.helper import (
    normalize_simple_list,
    canonicalize_platforms,
    to_int_or_none,
    make_hash8,
    now_iso_z,
    extract_game_id_from_item,
    site_url_contains_slug,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------
# Local utilities
# -----------------------
def _iso_now_z() -> str:
    return now_iso_z()


def _normalize_date(value: Optional[str]) -> Optional[str]:
    """Normalize date string to RFC3339 Z when possible; otherwise return None."""
    if not value:
        return None
    s = str(value).strip()
    try:
        # Accept ISO-like strings
        if "T" in s or s.endswith("Z") or "+" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    # Try some common date formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            continue
    return None


def _slugify(title: Optional[str]) -> str:
    if not title:
        return "untitled"
    s = str(title).lower()
    s = "".join(ch if (ch.isalnum() or ch in "- ") else "-" for ch in s)
    s = "-".join(s.split())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def _make_slug(title: Optional[str], release_date: Optional[str]) -> str:
    base = _slugify(title or "untitled")
    year = None
    if release_date:
        try:
            year = str(release_date).split("-")[0]
        except Exception:
            year = None
    if year and base.endswith(f"-{year}"):
        return base
    if year:
        return f"{base}-{year}"
    return base


def _deterministic_unified_id(slug: str, release_year: Optional[int], n: int = 8) -> str:
    y = "unknown" if release_year is None else str(release_year)
    base = f"{slug}-{y}"
    return f"{slug}-{y}-{make_hash8(base)[:n]}"


def _ensure_text_chunk(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a single serializable content doc with normalized metadata fields."""
    md = dict(metadata or {})
    if "unified_game_id" not in md:
        md["unified_game_id"] = md.get("unified_game_id") or None
    if "language" not in md or not md.get("language"):
        md["language"] = "en"
    # content_length: words count
    try:
        md["content_length"] = int(md.get("content_length") or len(text.split()))
    except Exception:
        md["content_length"] = len(text.split())
    # strip raw blob if present
    md.pop("raw_source_blob", None)
    return {"text": text, "metadata": md}


# -----------------------
# Per-source parsers
# -----------------------
def _parse_rawg(data: Any) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    results: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    items = data if isinstance(data, list) else [data] if isinstance(data, dict) and ("name" in data or "title" in data) else []
    for g in items:
        title = g.get("name") or g.get("title") or ""
        release_date = _normalize_date(g.get("released") or g.get("release_date"))
        slug = _make_slug(title, release_date)
        release_year = None
        if release_date and "-" in str(release_date):
            try:
                release_year = int(str(release_date).split("-")[0])
            except Exception:
                release_year = None
        unified = _deterministic_unified_id(slug, release_year)

        source_game_id = str(g.get("id")) if g.get("id") is not None else None
        game_id_int = to_int_or_none(g.get("id"))

        canonical = {
            "source": "rawg",
            "source_game_id": source_game_id,
            "game_id": game_id_int,
            "slug": slug,
            "title": title,
            "description": g.get("description") or g.get("description_raw") or "",
            "release_date": release_date,
            "release_year": release_year,
            "genres": normalize_simple_list(g.get("genres")),
            "platforms": canonicalize_platforms(normalize_simple_list(g.get("platforms"))),
            "developers": normalize_simple_list(g.get("developers")),
            "publishers": normalize_simple_list(g.get("publishers")),
            "tags": normalize_simple_list(g.get("tags")),
            "metacritic": g.get("metacritic"),
            "rating": g.get("rating"),
            "rating_count": to_int_or_none(g.get("ratings_count") or g.get("ratings")),
            "site_detail_url": g.get("website"),
            "created_at": _iso_now_z(),
            "raw_source_blob": json.dumps(g, ensure_ascii=False),
        }

        chunks: List[Dict[str, Any]] = []
        desc = canonical.get("description") or ""
        if desc.strip():
            text = f"Name: {title}\n\nDescription:\n{desc.strip()}"
            meta = {
                "source": "rawg",
                "source_game_id": source_game_id,
                "game_id": game_id_int,
                "unified_game_id": unified,
                "slug": slug,
                "title": title,
                "chunk_type": "description",
                "created_at": canonical["created_at"],
                "language": "en",
            }
            chunks.append(_ensure_text_chunk(text, meta))

        results.append(({"unified_game_id": unified, **canonical}, chunks))
    return results


def _parse_igdb(data: Any) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    results: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    items = data if isinstance(data, list) else [data] if isinstance(data, dict) and "name" in data else []
    for g in items:
        title = g.get("name") or ""
        release_raw = g.get("first_release_date") or g.get("release_date")
        rd = None
        try:
            if isinstance(release_raw, int):
                rd = datetime.utcfromtimestamp(release_raw).strftime("%Y-%m-%d")
            elif isinstance(release_raw, str):
                rd = release_raw.split("T")[0] if "T" in release_raw else release_raw
        except Exception:
            rd = None
        release_date = _normalize_date(rd)
        slug = _make_slug(title, release_date)
        release_year = None
        if release_date and "-" in str(release_date):
            try:
                release_year = int(str(release_date).split("-")[0])
            except Exception:
                release_year = None
        unified = _deterministic_unified_id(slug, release_year)

        source_game_id = str(g.get("id")) if g.get("id") is not None else None
        game_id_int = to_int_or_none(g.get("id"))

        canonical = {
            "source": "igdb",
            "source_game_id": source_game_id,
            "game_id": game_id_int,
            "slug": slug,
            "title": title,
            "description": g.get("description") or g.get("summary") or "",
            "summary": g.get("summary"),
            "storyline": g.get("storyline"),
            "release_date": release_date,
            "release_year": release_year,
            "genres": normalize_simple_list(g.get("genres")),
            "platforms": canonicalize_platforms(normalize_simple_list(g.get("platforms"))),
            "developers": normalize_simple_list(g.get("involved_companies")),
            "publishers": normalize_simple_list(g.get("publishers")),
            "themes": normalize_simple_list(g.get("themes")),
            "rating": g.get("rating"),
            "aggregated_rating": g.get("aggregated_rating"),
            "rating_count": to_int_or_none(g.get("rating_count")),
            "created_at": _iso_now_z(),
            "raw_source_blob": json.dumps(g, ensure_ascii=False),
        }

        chunks: List[Dict[str, Any]] = []
        body = canonical.get("description") or canonical.get("summary") or ""
        if body.strip():
            text = f"Name: {title}\n\nDescription:\n{body.strip()}"
            meta = {
                "source": "igdb",
                "source_game_id": source_game_id,
                "game_id": game_id_int,
                "unified_game_id": unified,
                "slug": slug,
                "title": title,
                "chunk_type": "description",
                "created_at": canonical["created_at"],
                "language": "en",
            }
            chunks.append(_ensure_text_chunk(text, meta))

        results.append(({"unified_game_id": unified, **canonical}, chunks))
    return results


def _parse_gamespot(data: Any) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Parse GameSpot structured dict into canonical + chunks.
    Filters Reviews and Articles to only those referencing the same game (by game_id or slug present in url).
    Dedupes items by id.
    """
    results: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    if isinstance(data, dict) and "Game Information" in data:
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    for entry in items:
        game_info = entry.get("Game Information", entry if isinstance(entry, dict) else {})
        title = game_info.get("name") or game_info.get("title") or ""
        release_date = _normalize_date(game_info.get("release_date") or game_info.get("released"))
        slug = _make_slug(title, release_date)
        release_year = None
        if release_date and "-" in str(release_date):
            try:
                release_year = int(str(release_date).split("-")[0])
            except Exception:
                release_year = None
        unified = _deterministic_unified_id(slug, release_year)

        source_game_id = str(game_info.get("id")) if game_info.get("id") is not None else None
        game_id_int = to_int_or_none(game_info.get("id"))

        canonical = {
            "source": "gamespot",
            "source_game_id": source_game_id,
            "game_id": game_id_int,
            "slug": slug,
            "title": title,
            "description": game_info.get("description") or game_info.get("deck") or "",
            "release_date": release_date,
            "release_year": release_year,
            "genres": normalize_simple_list(game_info.get("genres")),
            "themes": normalize_simple_list(game_info.get("themes")),
            "developers": normalize_simple_list(game_info.get("developers")),
            "publishers": normalize_simple_list(game_info.get("publishers")),
            "platforms": canonicalize_platforms(normalize_simple_list(game_info.get("platforms"))),
            "franchise": game_info.get("franchise"),
            "site_detail_url": game_info.get("site_detail_url"),
            "articles_count": len(entry.get("Articles") or []),
            "reviews_count": len(entry.get("Reviews") or []),
            "created_at": _iso_now_z(),
            "raw_source_blob": json.dumps(entry, ensure_ascii=False),
        }

        chunks: List[Dict[str, Any]] = []

        # main description chunk
        desc = canonical.get("description") or ""
        if desc.strip():
            t = f"Title: {title}\n\nDescription:\n{desc.strip()}"
            meta = {
                "source": "gamespot",
                "source_game_id": source_game_id,
                "game_id": game_id_int,
                "unified_game_id": unified,
                "slug": slug,
                "title": title,
                "chunk_type": "description",
                "site_detail_url": canonical.get("site_detail_url"),
                "created_at": canonical["created_at"],
                "language": "en",
            }
            chunks.append(_ensure_text_chunk(t, meta))

        # Reviews: filter by game association and dedupe by id
        seen_review_ids = set()
        for r in (entry.get("Reviews") or []):
            rid = r.get("id")
            if rid is not None and rid in seen_review_ids:
                continue
            # attempt extract game id from the review item
            r_game_id = extract_game_id_from_item(r)
            if r_game_id is not None:
                # require match
                if game_id_int is None or r_game_id != game_id_int:
                    continue
            else:
                # fallback to slug-in-url heuristic
                if not site_url_contains_slug(r.get("site_detail_url") or r.get("url"), slug):
                    continue

            seen_review_ids.add(rid)
            text_parts = []
            for fld in ("title", "deck", "body", "summary"):
                v = r.get(fld)
                if v:
                    text_parts.append(str(v).strip())
            text = " ".join(text_parts) if text_parts else json.dumps(r, ensure_ascii=False)
            meta = {
                "source": "gamespot",
                "source_game_id": source_game_id,
                "game_id": game_id_int,
                "unified_game_id": unified,
                "slug": slug,
                "title": title,
                "chunk_type": "review",
                "review_id": rid,
                "created_at": _iso_now_z(),
                "site_detail_url": r.get("site_detail_url") or r.get("url"),
                "language": "en",
            }
            chunks.append(_ensure_text_chunk(f"Review: {text}", meta))

        # Articles: filter & dedupe similarly
        seen_article_ids = set()
        for a in (entry.get("Articles") or []):
            aid = a.get("id")
            if aid is not None and aid in seen_article_ids:
                continue
            a_game_id = extract_game_id_from_item(a)
            if a_game_id is not None:
                if game_id_int is None or a_game_id != game_id_int:
                    continue
            else:
                if not site_url_contains_slug(a.get("site_detail_url") or a.get("url"), slug):
                    continue

            seen_article_ids.add(aid)
            text_parts = []
            for fld in ("title", "deck", "body", "summary"):
                v = a.get(fld)
                if v:
                    text_parts.append(str(v).strip())
            text = " ".join(text_parts) if text_parts else json.dumps(a, ensure_ascii=False)
            meta = {
                "source": "gamespot",
                "source_game_id": source_game_id,
                "game_id": game_id_int,
                "unified_game_id": unified,
                "slug": slug,
                "title": title,
                "chunk_type": "article",
                "article_id": aid,
                "created_at": _iso_now_z(),
                "site_detail_url": a.get("site_detail_url") or a.get("url"),
                "language": "en",
            }
            chunks.append(_ensure_text_chunk(f"Article: {text}", meta))

        # Releases: dedupe by id/platform/region; only keep releases with some relation to the game_info
        seen_release_keys = set()
        for rel in (entry.get("Releases") or []):
            rid = rel.get("id")
            platform_name = rel.get("platform") if isinstance(rel.get("platform"), str) else (rel.get("platform", {}).get("name") if isinstance(rel.get("platform"), dict) else "")
            region = rel.get("region") or ""
            key = (rid, platform_name, region)
            if key in seen_release_keys:
                continue
            seen_release_keys.add(key)
            rname = rel.get("name") or ""
            text = f"Release: {rname}\nPlatform: {platform_name}\nRegion: {region}"
            meta = {
                "source": "gamespot",
                "source_game_id": source_game_id,
                "game_id": game_id_int,
                "unified_game_id": unified,
                "slug": slug,
                "title": title,
                "chunk_type": "release",
                "release_id": rid,
                "created_at": _iso_now_z(),
                "language": "en",
            }
            chunks.append(_ensure_text_chunk(text, meta))

        results.append(({"unified_game_id": unified, **canonical}, chunks))

    return results


# -----------------------
# Public API
# -----------------------
def _load_json_input(source: Union[str, Dict[str, Any]]) -> Any:
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    return source


def load_and_prepare(source: Union[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load structured JSON from a path or pre-parsed dict, and return:
      - canonical_objs: List[Dict] (each canonical includes raw_source_blob JSON string)
      - content_docs: List[Dict] (plain serializable content docs with 'text' and 'metadata')
    """
    data = _load_json_input(source)
    canonical_objs: List[Dict[str, Any]] = []
    content_docs: List[Dict[str, Any]] = []

    # Determine parser heuristically
    if isinstance(data, dict):
        if "Game Information" in data or ("Games" in data and isinstance(data.get("Games"), list)):
            parsed = _parse_gamespot(data)
        elif "name" in data and ("metacritic" in data or "esrb_rating" in data):
            parsed = _parse_rawg(data)
        elif "name" in data:
            parsed = _parse_igdb(data)
        elif isinstance(data.get("results"), list):
            parsed = _parse_rawg(data.get("results"))
        else:
            parsed = _parse_gamespot(data) or _parse_rawg(data) or _parse_igdb(data)
    elif isinstance(data, list):
        first = data[0] if data else {}
        if "metacritic" in first or "esrb_rating" in first:
            parsed = _parse_rawg(data)
        elif "name" in first and "summary" in first:
            parsed = _parse_igdb(data)
        else:
            parsed = _parse_gamespot(data)
    else:
        raise ValueError("Unrecognized input data format for load_and_prepare")

    # parsed is list of (canonical, chunks)
    for canonical, docs in parsed:
        # enforce array types and numeric types
        for fld in ("genres", "platforms", "developers", "publishers", "tags", "themes"):
            if fld in canonical:
                canonical[fld] = canonical.get(fld) or []
        canonical["release_year"] = to_int_or_none(canonical.get("release_year"))
        canonical["game_id"] = to_int_or_none(canonical.get("game_id"))
        canonical["language"] = canonical.get("language") or "en"

        # ensure raw_source_blob is a JSON string (already provided by parsers)
        raw_blob = canonical.get("raw_source_blob")
        if raw_blob is None:
            canonical["raw_source_blob"] = json.dumps(canonical, ensure_ascii=False)
        elif not isinstance(raw_blob, str):
            canonical["raw_source_blob"] = json.dumps(raw_blob, ensure_ascii=False)

        canonical_objs.append(canonical)

        # attach unified_game_id and finalize chunk metadata
        for c in docs:
            md = c.get("metadata", {})
            if not md.get("unified_game_id"):
                md["unified_game_id"] = canonical.get("unified_game_id")
            if not md.get("language"):
                md["language"] = "en"
            # remove raw_source_blob from any chunk metadata (defensive)
            md.pop("raw_source_blob", None)
            # ensure content_length numeric
            try:
                md["content_length"] = int(md.get("content_length") or len((c.get("text") or "").split()))
            except Exception:
                md["content_length"] = len((c.get("text") or "").split())
            c["metadata"] = md
            content_docs.append(c)

    logger.info("Prepared %d canonical objects and %d content docs.", len(canonical_objs), len(content_docs))
    return canonical_objs, content_docs


# convenience wrapper (old code paths)
def load_documents(source: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    _, docs = load_and_prepare(source)
    return docs
