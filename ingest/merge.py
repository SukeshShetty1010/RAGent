#!/usr/bin/env python3
"""
merge.py

Pure merging utilities for game metadata from multiple sources (RAWG, IGDB, GameSpot).

This module exposes functions that operate on in-memory data structures (dicts/lists).
It does NOT perform file I/O or call any loader functions — that belongs in the runner.

Functions:
  - infer_schema(records) -> Dict[str, Set[str]]         # quick schema inference
  - normalize_record(record, source) -> dict            # per-source normalizer
  - merge_records(records_by_source, source_priority) -> dict
  - group_records(records, key_funcs) -> List[List[dict]] (optional)
  - validate_merged(merged) -> (bool, list[str])

Canonical merged shape returned by merge_records (high-level):
{
  "name": str,
  "slug": str,
  "release": {"date_unix": int|None, "date_iso": str|None, "source": str|None},
  "platforms": [...],
  "genres": [...],
  "descriptions": {"rawg": str|None, "igdb": str|None, "gamespot": str|None},
  "summaries": {...},
  "ratings": {...},
  "external_ids": {...},
  "related_text": {...},
  "images": {...},
  "sources": {...},   # provenance counts
  "merged_description": str|None
}
"""
from __future__ import annotations

import datetime
import re
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union


# -----------------------
# Utility helpers
# -----------------------
def safe_name(name: str) -> str:
    s = (name or "").strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", s) or "game"

def iso_from_unix(u: Optional[int]) -> Optional[str]:
    if not u:
        return None
    try:
        return datetime.datetime.utcfromtimestamp(int(u)).isoformat() + "Z"
    except Exception:
        return None

# -----------------------
# Schema inference
# -----------------------
def infer_schema(records: Iterable[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Quick, conservative schema inference.
    Returns dict: field -> set(types as strings)
    """
    types: Dict[str, Set[str]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for k, v in rec.items():
            t = type(v).__name__
            if k not in types:
                types[k] = set()
            types[k].add(t)
    return types

# -----------------------
# Normalizers (per-source)
# -----------------------
# These are intentionally conservative: they extract common fields while keeping
# an opaque 'raw' copy for inspection/provenance.

def normalize_rawg(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a RAWG record into the canonical partial schema."""
    r = {}
    r["name"] = rec.get("name")
    r["slug"] = rec.get("slug")
    r["released_str"] = rec.get("released")  # e.g., "2018-03-27"
    r["description_raw"] = rec.get("description_raw") or rec.get("description")
    r["summary"] = rec.get("short_description") or rec.get("deck")
    # platforms/genres on RAWG often are lists of dicts; leave as-is for downstream mapping
    r["platforms"] = rec.get("platforms")
    r["genres"] = rec.get("genres")
    r["rating"] = rec.get("rating")
    r["rating_count"] = rec.get("rating_count")
    r["rawg_id"] = rec.get("id")
    r["raw"] = rec
    return r

def normalize_igdb(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single IGDB record (or the first element of a wrapper)."""
    # IGDB wrapper (from your loader) can be either a record or a dict with 'clean'/'raw' arrays.
    if isinstance(rec, dict) and ("clean" in rec or "raw" in rec):
        # pick the first clean entry if present
        if rec.get("clean"):
            rec = rec["clean"][0] if isinstance(rec["clean"], list) and rec["clean"] else {}
        elif rec.get("raw"):
            rec = rec["raw"][0] if isinstance(rec["raw"], list) and rec["raw"] else {}

    r = {}
    r["name"] = rec.get("name")
    r["slug"] = rec.get("slug")
    r["first_release_date_unix"] = rec.get("first_release_date")
    r["summary"] = rec.get("summary")
    r["storyline"] = rec.get("storyline")
    # combine the best description candidate
    r["description_raw"] = rec.get("summary") or rec.get("storyline")
    r["platforms"] = rec.get("platforms")
    r["genres"] = rec.get("genres")
    r["screenshots"] = rec.get("screenshots")
    r["cover"] = rec.get("cover")
    r["igdb_id"] = rec.get("id")
    r["rating"] = rec.get("total_rating") or rec.get("aggregated_rating") or rec.get("rating")
    r["rating_count"] = rec.get("total_rating_count") or rec.get("aggregated_rating_count") or rec.get("rating_count")
    r["raw"] = rec
    return r

def normalize_gamespot(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a GameSpot entry. Accepts either the nested wrapper {game:..., related:...} or raw game dict."""
    rr = rec
    if isinstance(rec, dict) and "game" in rec:
        rr = rec["game"]
    r = {}
    r["name"] = rr.get("name") or rec.get("name")
    r["slug"] = rr.get("slug") or rec.get("slug")
    r["summary"] = rr.get("deck") or rec.get("deck")
    r["description_raw"] = rr.get("description") or rec.get("body") or rec.get("lede")
    r["gamespot_id"] = rr.get("id") or rec.get("id")
    r["related_text"] = rec.get("related") or rec.get("articles") or rec.get("related_text")
    r["raw"] = rec
    return r

# Dispatcher map
_NORMALIZERS = {
    "rawg": normalize_rawg,
    "igdb": normalize_igdb,
    "gamespot": normalize_gamespot,
}

def normalize_record(record: Any, source: str) -> Dict[str, Any]:
    """Generic normalizer that picks the appropriate per-source normalizer."""
    normalizer = _NORMALIZERS.get(source)
    if not normalizer:
        # if unknown source, return a shallow wrapper
        return {"name": (record.get("name") if isinstance(record, dict) else None), "raw": record}
    return normalizer(record)

# -----------------------
# Merge helpers
# -----------------------
def union_lists(a: Optional[List[Any]], b: Optional[List[Any]]) -> List[Any]:
    """Union two lists keeping order and preserving uniqueness."""
    a = a or []
    b = b or []
    seen = set()
    out = []
    for v in (a + b):
        try:
            key = v if not isinstance(v, dict) else jsonifiable_key(v)
        except Exception:
            key = str(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out

def jsonifiable_key(v: Any) -> str:
    """Create a stable key for basic JSON-able values (dict/list/primitive)."""
    # For dicts, prefer 'id'/'name' keys for identity if present
    if isinstance(v, dict):
        if "id" in v:
            return f"id:{v['id']}"
        if "name" in v:
            return f"name:{v['name']}"
    return str(v)

def choose_preferred(a: Any, b: Any, priority: List[str], a_src: str, b_src: str) -> Any:
    """Choose preferred scalar value when both exist, using source priority."""
    if a and not b:
        return a
    if b and not a:
        return b
    if not a and not b:
        return None
    # both non-empty -> compare priority index
    try:
        ai = priority.index(a_src)
    except ValueError:
        ai = len(priority)
    try:
        bi = priority.index(b_src)
    except ValueError:
        bi = len(priority)
    return a if ai <= bi else b

# -----------------------
# Main merge function
# -----------------------
def merge_records(records_by_source: Dict[str, Any], source_priority: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Merge multiple sources (in-memory) into a single canonical record dict.

    records_by_source is expected to be a dict like:
      { "rawg": [rec1, rec2, ...], "igdb": [rec1, ...] , "gamespot": [rec1, ...] }
    The function will:
      - pick representative record(s) from each source (first entry when lists)
      - normalize per-source
      - assemble canonical fields using priority-based tie-breakers
      - keep per-source descriptions and a concatenated merged_description
      - return canonical merged dict
    """
    if source_priority is None:
        source_priority = ["rawg", "igdb", "gamespot"]

    merged: Dict[str, Any] = {
        "name": None,
        "slug": None,
        "release": {"date_unix": None, "date_iso": None, "source": None},
        "platforms": [],
        "genres": [],
        "descriptions": {"rawg": None, "igdb": None, "gamespot": None},
        "summaries": {"rawg": None, "igdb": None, "gamespot": None},
        "ratings": {},
        "external_ids": {},
        "related_text": {},
        "images": {},
        "sources": {},
        "merged_description": None,
    }

    # Helper to grab first representative record from source value
    def representative(src_val: Any) -> Optional[Any]:
        if src_val is None:
            return None
        if isinstance(src_val, list):
            return src_val[0] if src_val else None
        if isinstance(src_val, dict):
            # loader sometimes returns wrapper dicts — accept them
            # if dict looks like { "clean": [...], "raw": [...] }, keep wrapper (normalizers handle it)
            return src_val
        return src_val

    # Normalize each source and populate merged fields
    for source in ("rawg", "igdb", "gamespot"):
        src_val = records_by_source.get(source)
        rep = representative(src_val)
        if not rep:
            merged["sources"][source] = {"count": 0}
            continue
        norm = normalize_record(rep, source)
        # provenance counts
        if isinstance(src_val, list):
            merged["sources"][source] = {"count": len(src_val)}
        elif isinstance(src_val, dict) and ("clean" in src_val or "records" in src_val):
            # wrapper with arrays inside
            cnt = 0
            if "clean" in src_val and isinstance(src_val["clean"], list):
                cnt = len(src_val["clean"])
            elif "records" in src_val and isinstance(src_val["records"], list):
                cnt = len(src_val["records"])
            merged["sources"][source] = {"count": cnt or 1}
        else:
            merged["sources"][source] = {"count": 1}

        # fill descriptions / summaries / images / ids / ratings
        merged["descriptions"][source] = norm.get("description_raw") or merged["descriptions"].get(source)
        merged["summaries"][source] = norm.get("summary") or norm.get("deck") or merged["summaries"].get(source)
        # images: keep a shallow mapping for each source where available
        img = {}
        if source == "rawg":
            raw = norm.get("raw", {}) or {}
            img["screenshots"] = raw.get("short_screenshots") or raw.get("screenshots")
            img["background_image"] = raw.get("background_image")
            merged["images"]["rawg"] = img
            merged["external_ids"]["rawg_id"] = norm.get("rawg_id")
            merged["ratings"]["rawg"] = {"rating": norm.get("rating"), "count": norm.get("rating_count")}
            # try to fill name/slug if not set
            merged["name"] = merged["name"] or norm.get("name")
            merged["slug"] = merged["slug"] or norm.get("slug")
        elif source == "igdb":
            merged["images"]["igdb"] = {"screenshots": norm.get("screenshots"), "cover": norm.get("cover")}
            merged["external_ids"]["igdb_id"] = norm.get("igdb_id")
            merged["ratings"]["igdb"] = {"rating": norm.get("rating"), "count": norm.get("rating_count")}
            if norm.get("first_release_date_unix"):
                merged["release"]["date_unix"] = merged["release"]["date_unix"] or norm.get("first_release_date_unix")
                merged["release"]["date_iso"] = merged["release"]["date_iso"] or iso_from_unix(norm.get("first_release_date_unix"))
                merged["release"]["source"] = merged["release"]["source"] or "igdb"
            merged["name"] = merged["name"] or norm.get("name")
            merged["slug"] = merged["slug"] or norm.get("slug")
            merged["platforms"] = union_lists(merged.get("platforms"), norm.get("platforms"))
            merged["genres"] = union_lists(merged.get("genres"), norm.get("genres"))
        elif source == "gamespot":
            merged["images"]["gamespot"] = {}  # GameSpot images often inside article bodies; keep raw under related_text
            merged["external_ids"]["gamespot_id"] = norm.get("gamespot_id")
            if norm.get("related_text"):
                merged["related_text"]["gamespot"] = norm.get("related_text")
            merged["name"] = merged["name"] or norm.get("name")
            merged["slug"] = merged["slug"] or norm.get("slug")
            merged["summaries"]["gamespot"] = merged["summaries"]["gamespot"] or norm.get("summary") or norm.get("deck")

    # If release.date_unix present, ensure iso is set
    if merged["release"]["date_unix"] and not merged["release"]["date_iso"]:
        merged["release"]["date_iso"] = iso_from_unix(merged["release"]["date_unix"])

    # Final canonical fallbacks
    if not merged["slug"] and merged["name"]:
        merged["slug"] = safe_name(merged["name"])

    # Build merged_description by concatenating available source descriptions (short-circuited)
    pieces = []
    for s in ("rawg", "igdb", "gamespot"):
        d = merged["descriptions"].get(s)
        if d and isinstance(d, str) and d.strip():
            pieces.append(f"[{s}] {d.strip()}")
    merged["merged_description"] = "\n\n".join(pieces) if pieces else None

    return merged

# -----------------------
# Optional grouping & dedupe helpers (small heuristics)
# -----------------------
def simple_name_key(rec: Dict[str, Any]) -> str:
    """Simple normalization key for grouping based on name/slug."""
    name = rec.get("name") or rec.get("slug") or ""
    return safe_name(name)

def group_records(records: List[Dict[str, Any]], key_func: Callable[[Dict[str, Any]], str] = simple_name_key) -> List[List[Dict[str, Any]]]:
    """
    Group records by a simple key function. Useful if you plan to merge multiple separate records
    (e.g., many IGDB/rawg results) into single canonical entries.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        k = key_func(r)
        groups.setdefault(k, []).append(r)
    return list(groups.values())

# -----------------------
# Validation
# -----------------------
def validate_merged(merged: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Basic validation of the merged record. Returns (is_valid, warnings_list).
    """
    warnings = []
    if not merged.get("name"):
        warnings.append("name is missing")
    # release date is optional; warn if missing
    if not merged.get("release", {}).get("date_unix") and not merged.get("release", {}).get("date_iso"):
        warnings.append("release date missing")
    # ensure at least one source contributed
    if not any((v.get("count", 0) for v in merged.get("sources", {}).values())):
        warnings.append("no sources contributed data")
    return (len(warnings) == 0, warnings)
