# ingest/merge.py
"""
Merge per-source canonical objects into a single canonical object.

Inputs:
  - canonical_objs: List[Dict] where each dict is a per-source canonical produced by ingest.loader.load_and_prepare()

Output:
  - merged canonical dict (plain JSON-serializable)
    - unified_game_id (deterministic slug-year-8hex)
    - doc_type = "canonical"
    - merge_time (now)
    - raw_source_blob : JSON string mapping {source: raw_json_string}
    - arrays as native lists (genres/platforms/developers/publishers/tags/themes)
    - game_id as int or None
    - release_year as int or None
    - deduped Releases/Articles/Reviews lists (if present across sources)
"""
from typing import List, Dict, Any, Optional
import json
from collections import OrderedDict

from data.helper import (
    normalize_simple_list,
    canonicalize_platforms,
    to_int_or_none,
    make_hash8,
    now_iso_z,
)

# source precedence when picking scalar fields
SOURCE_PRECEDENCE = ["gamespot", "igdb", "rawg"]


def _choose_by_precedence(objs: List[Dict[str, Any]], key: str) -> Optional[Any]:
    for src in SOURCE_PRECEDENCE:
        for o in objs:
            if o.get("source") == src and o.get(key) not in (None, "", [], {}):
                return o.get(key)
    for o in objs:
        if o.get(key) not in (None, "", [], {}):
            return o.get(key)
    return None


def _union_lists_preserve_case(list_of_lists: List[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in list_of_lists:
        if not lst:
            continue
        for v in lst:
            if v is None:
                continue
            s = str(v).strip()
            if s == "":
                continue
            k = s.lower()
            if k not in seen:
                seen.add(k)
                out.append(s)
    return out


def _longest_text(objs: List[Dict[str, Any]], keys: List[str]) -> Optional[str]:
    best = ""
    for o in objs:
        for k in keys:
            v = o.get(k)
            if isinstance(v, str) and len(v) > len(best):
                best = v
    return best or None


def _collect_and_dedupe_records(objs: List[Dict[str, Any]], field_name: str) -> List[Dict[str, Any]]:
    """
    Collect lists of dict-like records from objs[field_name] and dedupe by 'id' where possible.
    Returns list of representative dicts (first-seen).
    If items are strings, they are preserved (deduped).
    """
    seen_ids = set()
    seen_texts = set()
    out: List[Dict[str, Any]] = []

    for o in objs:
        items = o.get(field_name) or []
        if not items:
            continue
        if isinstance(items, dict):
            items = [items]
        for it in items:
            # if dict and has id, use id dedupe
            if isinstance(it, dict):
                iid = it.get("id")
                if iid is not None:
                    key = f"id::{iid}"
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    out.append(it)
                    continue
                # no id - try to stringify a canonical text representation
                text = it.get("title") or it.get("name") or json.dumps(it, sort_keys=True, ensure_ascii=False)
                key = f"text::{text}"
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                out.append(it)
            else:
                # primitive value (str/int); dedupe by string repr
                t = str(it)
                if t in seen_texts:
                    continue
                seen_texts.add(t)
                out.append({"value": t})
    return out


def merge_canonical_objects(canonical_objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not canonical_objs:
        raise ValueError("merge_canonical_objects requires at least one canonical object")

    # choose base record for defaults
    base = sorted(canonical_objs, key=lambda x: len(x.keys()), reverse=True)[0]

    slug = _choose_by_precedence(canonical_objs, "slug") or base.get("slug") or (base.get("title") or "").lower().replace(" ", "-")
    title = _choose_by_precedence(canonical_objs, "title") or base.get("title") or ""
    # release_year int or None
    ry = _choose_by_precedence(canonical_objs, "release_year")
    if ry is None:
        rd = _choose_by_precedence(canonical_objs, "release_date")
        if isinstance(rd, str) and len(rd) >= 4:
            try:
                ry = int(str(rd)[:4])
            except Exception:
                ry = None
    release_year = to_int_or_none(ry)

    # deterministic unified id: slug-year-8hex
    base_for_hash = f"{slug}-{release_year or 'unknown'}"
    suffix = make_hash8(base_for_hash)
    unified_game_id = f"{slug}-{release_year or 'unknown'}-{suffix}"

    merged: Dict[str, Any] = {
        "unified_game_id": unified_game_id,
        "slug": slug,
        "title": title,
        "doc_type": "canonical",
        "merge_time": now_iso_z(),
    }

    # choose longest description/summary/storyline
    desc = _longest_text(canonical_objs, ["description", "summary", "storyline"])
    if desc:
        merged["description"] = desc

    # arrays: merge & dedupe
    merged["genres"] = _union_lists_preserve_case([normalize_simple_list(o.get("genres")) for o in canonical_objs])
    merged["platforms"] = canonicalize_platforms(_union_lists_preserve_case([normalize_simple_list(o.get("platforms")) for o in canonical_objs]))
    merged["developers"] = _union_lists_preserve_case([normalize_simple_list(o.get("developers")) for o in canonical_objs])
    merged["publishers"] = _union_lists_preserve_case([normalize_simple_list(o.get("publishers")) for o in canonical_objs])
    merged["tags"] = _union_lists_preserve_case([normalize_simple_list(o.get("tags")) for o in canonical_objs])
    merged["themes"] = _union_lists_preserve_case([normalize_simple_list(o.get("themes")) for o in canonical_objs])

    # numeric aggregation for ratings
    ratings: List[float] = []
    for o in canonical_objs:
        for k in ("rating", "aggregated_rating", "metacritic", "total_rating"):
            v = o.get(k)
            if v is None:
                continue
            try:
                ratings.append(float(v))
            except Exception:
                continue
    if ratings:
        merged["score_normalized"] = float(sum(ratings) / len(ratings))
        merged["score_count"] = len(ratings)

    # rating_count
    rc = _choose_by_precedence(canonical_objs, "rating_count")
    merged["rating_count"] = to_int_or_none(rc)

    # release_date/year, site_detail_url, franchise
    rd = _choose_by_precedence(canonical_objs, "release_date")
    if rd:
        merged["release_date"] = rd
    merged["release_year"] = release_year

    sd = _choose_by_precedence(canonical_objs, "site_detail_url")
    if sd:
        merged["site_detail_url"] = sd
    fr = _choose_by_precedence(canonical_objs, "franchise")
    if fr:
        merged["franchise"] = fr

    # collect sources and source_game_ids + raw blobs
    sources: List[str] = []
    source_game_ids: Dict[str, str] = {}
    raw_map: Dict[str, str] = {}

    for o in canonical_objs:
        src = o.get("source") or "unknown"
        if src not in sources:
            sources.append(src)
        sid = o.get("source_game_id")
        if sid is not None:
            source_game_ids[src] = str(sid)
        # raw_source_blob: ensure string; loader stores a JSON string already but accept dicts too
        rb = o.get("raw_source_blob")
        if rb is None:
            # fallback: serialize the object itself
            raw_map[src] = json.dumps(o, ensure_ascii=False)
        else:
            raw_map[src] = rb if isinstance(rb, str) else json.dumps(rb, ensure_ascii=False)

    merged["sources"] = sources
    merged["source_game_ids"] = source_game_ids
    # Store raw_source_blob as a JSON string mapping {source: raw_json_string} (per your requirement)
    merged["raw_source_blob"] = json.dumps(raw_map, ensure_ascii=False)

    # game_id: first non-null integer across sources
    gid = None
    for o in canonical_objs:
        cand = o.get("game_id")
        if cand is not None:
            g = to_int_or_none(cand)
            if g is not None:
                gid = g
                break
    merged["game_id"] = gid

    # created_at: earliest if available else now
    created_vals = [o.get("created_at") for o in canonical_objs if o.get("created_at")]
    merged["created_at"] = min(created_vals) if created_vals else now_iso_z()

    # Collect and dedupe Releases / Articles / Reviews if present across inputs
    # These will be stored as native lists of dicts (representative items) under canonical if any exist
    for coll_name in ("Releases", "Articles", "Reviews"):
        collected = _collect_and_dedupe_records(canonical_objs, coll_name)
        if collected:
            merged[coll_name] = collected

    return merged
