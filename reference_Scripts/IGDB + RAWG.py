# rough.py (updated)
from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, Field, AnyUrl
# conditional import for validator compatibility (pydantic v1 vs v2)
try:
    from pydantic import field_validator  # v2 style
    _HAS_FIELD_VALIDATOR = True
except Exception:
    from pydantic import validator  # v1 style (deprecated in v2)
    _HAS_FIELD_VALIDATOR = False

import json
import pathlib
import re

# ------------------------
# Pydantic schema for merged record
# ------------------------
class Ratings(BaseModel):
    rawg: Optional[float] = None
    igdb: Optional[float] = None
    metacritic: Optional[int] = None
    # Keep original full-rating objects as provenance if available
    rawg_detail: Optional[Dict[str, Any]] = None
    igdb_detail: Optional[Dict[str, Any]] = None


class SourceProvenance(BaseModel):
    rawg: Optional[Dict[str, Any]] = None
    igdb: Optional[Dict[str, Any]] = None


class GameMerged(BaseModel):
    # canonical top-level fields
    unified_id: Optional[str] = None
    title: str
    slug: Optional[str] = None
    description: Optional[str] = None
    release_date: Optional[date] = None

    # ids per source
    rawg_id: Optional[int] = None
    igdb_id: Optional[int] = None

    platforms: List[str] = Field(default_factory=list)
    genres: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    ratings: Ratings = Field(default_factory=Ratings)
    urls: List[AnyUrl] = Field(default_factory=list)

    # raw source preserved for provenance and debugging
    source: SourceProvenance = Field(default_factory=SourceProvenance)

    merged_from: Optional[Dict[str, Any]] = None

    # conditional validator compatible with Pydantic v1 & v2
    if _HAS_FIELD_VALIDATOR:
        @field_validator("title")
        @classmethod
        def _title_validator(cls, v):
            if not v or not str(v).strip():
                raise ValueError("title must be a non-empty string")
            return str(v).strip()
    else:
        @validator("title")
        @classmethod
        def _title_validator(cls, v):
            if not v or not str(v).strip():
                raise ValueError("title must be a non-empty string")
            return str(v).strip()


# ------------------------
# Utility helpers
# ------------------------
def load_json(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(data: Dict[str, Any], path: str) -> None:
    """Saves a dictionary to a JSON file with indentation."""
    p = pathlib.Path(path)
    # default=str helps if any date objects slipped through without conversion
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _iso_date_from_rawg(rawg_date: Optional[str]) -> Optional[date]:
    """RAWG uses YYYY-MM-DD in `released` typically. Return date object or None."""
    if not rawg_date:
        return None
    try:
        # Accept YYYY-MM-DD or YYYY
        dt = datetime.fromisoformat(rawg_date)
        return dt.date()
    except Exception:
        # Try partial parse
        try:
            return datetime.strptime(rawg_date, "%Y-%m-%d").date()
        except Exception:
            return None


def _date_from_unix(ts: Optional[int]) -> Optional[date]:
    if not ts:
        return None
    try:
        return datetime.utcfromtimestamp(int(ts)).date()
    except Exception:
        return None


def _list_union_normalize(a: Optional[List[Any]], b: Optional[List[Any]]) -> List[str]:
    """Union two lists of strings/objects; when objects, try to extract `name` key."""
    out = []
    if a:
        for item in a:
            name = _extract_name_from_item(item)
            if name:
                out.append(name)
    if b:
        for item in b:
            name = _extract_name_from_item(item)
            if name:
                out.append(name)
    # normalize & dedupe preserving first-seen capitalization
    seen = set()
    result = []
    for v in out:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(v.strip())
    return result

def _extract_name_from_item(item: Any) -> Optional[str]:
    """Extract a human-readable name from various possible item shapes.

    Returns None for numeric-only ids (we don't want '48' leaking into platforms list).
    """
    if item is None:
        return None
    # dicts with 'name' or nested 'platform'/'genre' objects
    if isinstance(item, dict):
        # RAWG platform object: { "platform": { "id": 4, "name": "PC", ... }, ... }
        if item.get("name") and isinstance(item.get("name"), str):
            return item.get("name").strip()
        if item.get("platform") and isinstance(item.get("platform"), dict) and item["platform"].get("name"):
            return item["platform"]["name"].strip()
        if item.get("genre") and isinstance(item.get("genre"), dict) and item["genre"].get("name"):
            return item["genre"]["name"].strip()
        if item.get("url") and isinstance(item.get("url"), str):
            # not a name, skip
            return None
        # fallback: try slug
        if item.get("slug") and isinstance(item.get("slug"), str):
            slug = item.get("slug").strip()
            if not slug.isdigit():
                return slug
        return None
    # strings that are not purely numeric are OK
    if isinstance(item, str):
        s = item.strip()
        if s and not re.fullmatch(r"\d+", s):
            return s
        return None
    # ints / floats => numeric id, return None (do not include)
    return None


# ------------------------
# Helpers to unwrap wrappers & detect content types
# ------------------------
def _looks_like_rawg(rec: Any) -> bool:
    """Heuristic to decide whether a dict looks like a RAWG record."""
    if not isinstance(rec, dict):
        return False
    if rec.get("description_raw") or rec.get("released") or rec.get("metacritic"):
        return True
    # RAWG often has platforms as list of dicts with a 'platform' key
    platforms = rec.get("platforms")
    if isinstance(platforms, list) and any(isinstance(p, dict) and p.get("platform") for p in platforms):
        return True
    return False

def _looks_like_igdb(wrapper: Any) -> bool:
    """Heuristic to decide whether wrapper is IGDB style (has records[0].clean)."""
    if not isinstance(wrapper, dict):
        return False
    recs = wrapper.get("records")
    if isinstance(recs, list) and recs:
        first = recs[0]
        if isinstance(first, dict) and isinstance(first.get("clean"), list):
            return True
        # IGDB records may have aggregated_rating or first_release_date on first-level
        if isinstance(first, dict) and (first.get("aggregated_rating") or first.get("first_release_date")):
            return True
    # fallback: presence of numeric genre ids (list of ints) suggests IGDB un-resolved ids
    if isinstance(wrapper.get("genres"), list) and wrapper.get("genres") and all(isinstance(x, int) for x in wrapper.get("genres")):
        return True
    return False

def _get_rawg_record(rawg_wrapper: Dict[str, Any]) -> Dict[str, Any]:
    """Return the best candidate RAWG record from a wrapper or return the input if it already is a record."""
    if not rawg_wrapper:
        return {}
    # If the wrapper contains an array of records, pick the first that looks like RAWG
    if isinstance(rawg_wrapper, dict) and "records" in rawg_wrapper and isinstance(rawg_wrapper["records"], list) and rawg_wrapper["records"]:
        for candidate in rawg_wrapper["records"]:
            if _looks_like_rawg(candidate):
                return candidate
        return rawg_wrapper["records"][0]
    # If this dict itself looks like RAWG, return it
    if _looks_like_rawg(rawg_wrapper):
        return rawg_wrapper
    # fallback: return as-is
    return rawg_wrapper

def choose_igdb_clean_record(igdb_wrapper: Dict[str, Any], rawg_name: str = "", rawg_slug: str = "") -> Dict[str, Any]:
    """Choose the best IGDB 'clean' record corresponding to the RAWG game.

    Heuristics:
    1) exact name match (case-insensitive)
    2) exact slug match
    3) substring name match
    4) highest aggregated_rating / total_rating fallback
    """
    if not igdb_wrapper:
        return {}
    clean_list = []
    if isinstance(igdb_wrapper, dict) and isinstance(igdb_wrapper.get("records"), list) and igdb_wrapper["records"]:
        first = igdb_wrapper["records"][0]
        if isinstance(first, dict) and isinstance(first.get("clean"), list):
            clean_list = first["clean"]
        else:
            # records may already be clean records
            clean_list = igdb_wrapper["records"]
    elif isinstance(igdb_wrapper, list):
        clean_list = igdb_wrapper

    if not clean_list:
        return {}

    rawg_name_l = (rawg_name or "").strip().lower()
    rawg_slug_l = (rawg_slug or "").strip().lower()

    # 1) exact name match
    for r in clean_list:
        if isinstance(r, dict) and r.get("name") and r["name"].strip().lower() == rawg_name_l:
            return r
    # 2) exact slug match
    for r in clean_list:
        if isinstance(r, dict) and r.get("slug") and r["slug"].strip().lower() == rawg_slug_l:
            return r
    # 3) substring match
    for r in clean_list:
        if isinstance(r, dict) and r.get("name") and rawg_name_l and rawg_name_l in r["name"].strip().lower():
            return r
    # 4) pick highest scored record
    def score(r: Dict[str, Any]) -> float:
        if not isinstance(r, dict):
            return 0.0
        return float(r.get("aggregated_rating") or r.get("total_rating") or r.get("rating") or 0.0)
    best = max(clean_list, key=score)
    return best if isinstance(best, dict) else {}

def _get_igdb_record(igdb_wrapper: Dict[str, Any], rawg_rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a canonical IGDB cleaned record from the wrapper using selection heuristics."""
    if not igdb_wrapper:
        return {}
    # If wrapper includes a 'clean' list, choose the best matching clean record
    igdb_choice = choose_igdb_clean_record(igdb_wrapper, rawg_rec.get("name", ""), rawg_rec.get("slug", ""))
    if igdb_choice:
        return igdb_choice
    # fallback: pick first record if looks like an IGDB record
    if isinstance(igdb_wrapper, dict) and isinstance(igdb_wrapper.get("records"), list) and igdb_wrapper["records"]:
        first = igdb_wrapper["records"][0]
        # if first is a wrapper with .clean handled earlier, else if first itself is game record return first
        if isinstance(first, dict) and not isinstance(first.get("clean"), list):
            return first
    # fallback to returning wrapper unchanged
    return igdb_wrapper


# ------------------------
# Lightweight merge implementation (fixed)
# ------------------------
def merge_records(rawg: Dict[str, Any], igdb: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two source dictionaries into a canonical dict that fits GameMerged.

    This implementation:
    - detects which wrapper contains RAWG-style vs IGDB-style data
    - unwraps wrapper(s) and picks the correct IGDB clean record matching RAWG
    - normalizes lists to drop numeric-only ids
    - properly prefixes unified_id based on which id is used
    """
    merged: Dict[str, Any] = {}

    # preserve raw wrappers for provenance
    merged["source"] = {"rawg": rawg, "igdb": igdb}

    # Detect if wrappers are swapped or which contains RAWG/IGDB
    # Prefer to find RAWG-like content inside either wrapper
    rawg_candidate = None
    igdb_candidate = None

    # First try to find RAWG-style content
    if _looks_like_rawg(rawg):
        rawg_candidate = rawg
    elif _looks_like_rawg(igdb):
        rawg_candidate = igdb

    # Find IGDB-style wrapper
    if _looks_like_igdb(igdb):
        igdb_candidate = igdb
    elif _looks_like_igdb(rawg):
        igdb_candidate = rawg

    # If detection failed, fallback to given labels
    if rawg_candidate is None:
        rawg_candidate = rawg
    if igdb_candidate is None:
        igdb_candidate = igdb

    # Unwrap to actual record objects
    rawg_rec = _get_rawg_record(rawg_candidate)
    igdb_rec = _get_igdb_record(igdb_candidate, rawg_rec)

    # IDs
    merged["rawg_id"] = rawg_rec.get("id")
    merged["igdb_id"] = igdb_rec.get("id")
    # Build unified_id from the actual ID we have (prefer RAWG if present)
    if merged["rawg_id"]:
        merged["unified_id"] = f"rawg:{merged['rawg_id']}"
    elif merged["igdb_id"]:
        merged["unified_id"] = f"igdb:{merged['igdb_id']}"
    else:
        merged["unified_id"] = None

    # Title / slug (prefer RAWG title but keep fallback to IGDB)
    title = (
        rawg_rec.get("name_original")
        or rawg_rec.get("name")
        or igdb_rec.get("name")
        or igdb_rec.get("resolved_name")
        or igdb_rec.get("slug")
        or rawg_rec.get("slug")
        or ""
    )
    merged["title"] = title.strip() if isinstance(title, str) else str(title)

    merged["slug"] = (igdb_rec.get("slug") or rawg_rec.get("slug") or None)

    # Description (prefer RAWG description_raw -> RAWG description -> IGDB summary -> IGDB storyline)
    desc = None
    if isinstance(rawg_rec, dict):
        desc = rawg_rec.get("description_raw") or rawg_rec.get("description")
    if not desc:
        desc = igdb_rec.get("summary") or igdb_rec.get("storyline")
    merged["description"] = desc.strip() if isinstance(desc, str) else None

    # Release date: prefer RAWG 'released' (ISO string), else IGDB unix ts
    rd = _iso_date_from_rawg(rawg_rec.get("released")) if isinstance(rawg_rec, dict) else None
    if not rd:
        rd = _date_from_unix(igdb_rec.get("first_release_date"))
    merged["release_date"] = rd.isoformat() if rd else None

    # Platforms, genres, tags: union and dedupe, with object-aware extraction
    merged["platforms"] = _list_union_normalize(rawg_rec.get("platforms") if isinstance(rawg_rec, dict) else None,
                                                igdb_rec.get("platforms") if isinstance(igdb_rec, dict) else None)
    merged["genres"] = _list_union_normalize(rawg_rec.get("genres") if isinstance(rawg_rec, dict) else None,
                                             igdb_rec.get("genres") if isinstance(igdb_rec, dict) else None)
    merged["tags"] = _list_union_normalize(rawg_rec.get("tags") if isinstance(rawg_rec, dict) else None,
                                           igdb_rec.get("tags") if isinstance(igdb_rec, dict) else None)

    # Ratings
    merged["ratings"] = {
        "rawg": rawg_rec.get("rating") if isinstance(rawg_rec, dict) else None,
        "igdb": (igdb_rec.get("aggregated_rating") or igdb_rec.get("total_rating") or igdb_rec.get("rating"))
                if isinstance(igdb_rec, dict) else None,
        "metacritic": rawg_rec.get("metacritic") if isinstance(rawg_rec, dict) else None,
        "rawg_detail": rawg_rec.get("ratings") if isinstance(rawg_rec, dict) else None,
        "igdb_detail": {k: igdb_rec.get(k) for k in ["aggregated_rating", "total_rating", "rating", "rating_count", "total_rating_count"] if isinstance(igdb_rec, dict) and igdb_rec.get(k) is not None}
    }

    # URLs collection
    urls = set()
    if isinstance(rawg_rec, dict):
        if rawg_rec.get("website"):
            urls.add(rawg_rec.get("website"))
        if rawg_rec.get("metacritic_url"):
            urls.add(rawg_rec.get("metacritic_url"))
        for s in (rawg_rec.get("stores") or []):
            if isinstance(s, dict) and s.get("url"):
                urls.add(s.get("url"))
    if isinstance(igdb_rec, dict):
        if igdb_rec.get("url"):
            urls.add(igdb_rec.get("url"))
        # igdb may store `websites` as list of dicts or strings
        for w in (igdb_rec.get("websites") or []):
            if isinstance(w, dict) and w.get("url"):
                urls.add(w.get("url"))
            elif isinstance(w, str):
                urls.add(w)
    # filter out None and empty strings
    merged["urls"] = [u for u in list(urls) if u]

    # merged_from: compute after unwrapping so decision is correct
    merged["merged_from"] = {
        "title_from": "rawg" if (isinstance(rawg_rec, dict) and (rawg_rec.get("name") or rawg_rec.get("name_original"))) else "igdb",
        "description_from": "rawg" if (isinstance(rawg_rec, dict) and (rawg_rec.get("description_raw") or rawg_rec.get("description"))) else "igdb",
    }

    return merged


# ------------------------
# Unit tests (pytest)
# ------------------------
def test_merge_and_schema_validation():
    """Load the two sample JSONs and ensure the merged object conforms to GameMerged model."""
    # Adjust paths if running locally without these specific /mnt paths
    try:
        rawg_path = "/mnt/data/far_cry_5_rawg.json"
        igdb_path = "/mnt/data/far_cry_5_igdb.json"
        rawg = load_json(rawg_path)
        igdb = load_json(igdb_path)
    except FileNotFoundError:
        # Fallback for local testing if files are in CWD
        rawg = load_json("far_cry_5_rawg.json")
        igdb = load_json("far_cry_5_igdb.json")

    merged_dict = merge_records(rawg, igdb)

    # Basic sanity asserts before Pydantic validation
    assert merged_dict.get("title"), "Merged title must be present"
    # parse and validate
    gm = GameMerged.parse_obj({
        "unified_id": merged_dict.get("unified_id"),
        "title": merged_dict.get("title"),
        "slug": merged_dict.get("slug"),
        "description": merged_dict.get("description"),
        "release_date": merged_dict.get("release_date"),
        "rawg_id": merged_dict.get("rawg_id"),
        "igdb_id": merged_dict.get("igdb_id"),
        "platforms": merged_dict.get("platforms"),
        "genres": merged_dict.get("genres"),
        "tags": merged_dict.get("tags"),
        "ratings": merged_dict.get("ratings"),
        "urls": merged_dict.get("urls"),
        "source": merged_dict.get("source"),
        "merged_from": merged_dict.get("merged_from"),
    })

    # Validate a few expectations
    assert gm.title.lower().startswith("far cry") or "far cry" in gm.title.lower()
    
    if gm.release_date:
        assert isinstance(gm.release_date, date)

    # Ratings keeps per-source keys
    assert hasattr(gm.ratings, "rawg")
    assert hasattr(gm.ratings, "igdb")


def test_key_merge_tactics_applied():
    """Test that preferred sources were used for title/description when present."""
    try:
        rawg = load_json("/mnt/data/far_cry_5_rawg.json")
        igdb = load_json("/mnt/data/far_cry_5_igdb.json")
    except FileNotFoundError:
        rawg = load_json("far_cry_5_rawg.json")
        igdb = load_json("far_cry_5_igdb.json")

    merged = merge_records(rawg, igdb)

    # If RAWG provided description_raw we should have taken it
    # But because the wrapper may be present, inspect rawg wrapper for description_raw if top-level missing
    rawg_rec = _get_rawg_record(rawg)
    if isinstance(rawg_rec, dict) and rawg_rec.get("description_raw"):
        assert merged.get("description") == rawg_rec.get("description_raw").strip()

    # Slug should prefer IGDB if present
    igdb_rec = _get_igdb_record(igdb, rawg_rec)
    if isinstance(igdb_rec, dict) and igdb_rec.get("slug"):
        assert merged.get("slug") == igdb_rec.get("slug")

    # Platforms should be a list
    assert isinstance(merged.get("platforms"), list)


# If module run directly, execute merge and save file
if __name__ == "__main__":
    try:
        # Ensure these files exist in your directory (fallback order preserved)
        try:
            rawg = load_json("/mnt/data/far_cry_5_rawg.json")
            igdb = load_json("/mnt/data/far_cry_5_igdb.json")
        except FileNotFoundError:
            rawg = load_json("far_cry_5_rawg.json")
            igdb = load_json("far_cry_5_igdb.json")
        
        merged = merge_records(rawg, igdb)
        
        output_filename = "merged_far_cry_5.json"
        save_json(merged, output_filename)
        
        print(f"Success! Merged data saved to: {output_filename}")
        
    except Exception as e:
        print(f"Error: {e}")
