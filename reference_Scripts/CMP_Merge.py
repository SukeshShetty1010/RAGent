# merge_three_sources.py
from __future__ import annotations
import json
import pathlib
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, date
from pydantic import BaseModel, Field, AnyUrl

# reuse helpers & parts of your rough.py design (validators + normalization)
try:
    from pydantic import field_validator
    _HAS_FIELD_VALIDATOR = True
except Exception:
    from pydantic import validator
    _HAS_FIELD_VALIDATOR = False

# ---------- Pydantic schema (extended) ----------
class Ratings(BaseModel):
    rawg: Optional[float] = None
    igdb: Optional[float] = None
    metacritic: Optional[int] = None
    rawg_detail: Optional[Dict[str, Any] | List[Any]] = None
    igdb_detail: Optional[Dict[str, Any] | List[Any]] = None
    normalized_0_100: Optional[int] = None


class SourceProvenance(BaseModel):
    rawg: Optional[Dict[str, Any]] = None
    igdb: Optional[Dict[str, Any]] = None
    gamespot: Optional[Dict[str, Any]] = None


class GameMerged(BaseModel):
    # canonical top-level fields
    unified_id: Optional[str] = None
    title: str
    slug: Optional[str] = None
    description: Optional[str] = None
    release_date: Optional[date] = None
    release_year: Optional[int] = None
    release_dates: List[Dict[str, Optional[str]]] = Field(default_factory=list)

    # ids per source
    rawg_id: Optional[int] = None
    igdb_id: Optional[int] = None
    gamespot_id: Optional[str] = None

    # classification
    platforms: List[str] = Field(default_factory=list)
    genres: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    developers: List[str] = Field(default_factory=list)
    publishers: List[str] = Field(default_factory=list)

    # age / rating classification
    age_ratings: List[str] = Field(default_factory=list)
    esrb_rating: Optional[str] = None

    # ratings & urls
    ratings: Ratings = Field(default_factory=Ratings)
    urls: List[AnyUrl] = Field(default_factory=list)
    stores: List[str] = Field(default_factory=list)
    websites: List[AnyUrl] = Field(default_factory=list)

    # gamespot textual & raw provenance
    gamespot: Dict[str, Any] = Field(default_factory=dict)

    # flattened documents for RAG ingestion
    documents: List[Dict[str, Any]] = Field(default_factory=list)

    # provenance
    source: SourceProvenance = Field(default_factory=SourceProvenance)
    merged_from: Optional[Dict[str, Any]] = None

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


# ---------- Utilities (based on your rough.py helpers) ----------
def load_json(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(data: Any, path: str) -> None:
    p = pathlib.Path(path)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _iso_date_from_rawg(rawg_date: Optional[str]) -> Optional[date]:
    if not rawg_date:
        return None
    try:
        return datetime.fromisoformat(rawg_date).date()
    except Exception:
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


def _extract_name_from_item(item: Any) -> Optional[str]:
    if item is None:
        return None
    if isinstance(item, dict):
        if item.get("name") and isinstance(item.get("name"), str):
            return item.get("name").strip()
        if item.get("platform") and isinstance(item["platform"], dict) and item["platform"].get("name"):
            return item["platform"]["name"].strip()
        if item.get("company") and isinstance(item["company"], dict) and item["company"].get("name"):
            return item["company"]["name"].strip()
        if item.get("url") and isinstance(item.get("url"), str):
            return None
        if item.get("slug") and isinstance(item.get("slug"), str):
            slug = item.get("slug").strip()
            if not slug.isdigit():
                return slug
        return None
    if isinstance(item, str):
        s = item.strip()
        if s and not re.fullmatch(r"\d+", s):
            return s
        return None
    return None


def _list_union_normalize(*lists: Optional[List[Any]]) -> List[str]:
    out = []
    for a in lists:
        if a:
            for item in a:
                name = _extract_name_from_item(item)
                if name:
                    out.append(name)
    seen = set()
    result = []
    for v in out:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(v.strip())
    return result


# ---------- GameSpot helpers ----------
def _collect_gamespot_buckets(gamespot_wrapper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract normalized gamespot structure:
      - game_info: pruned game object
      - articles: list of normalized article objects
      - reviews: list of normalized review objects
      - releases: list of normalized release objects
    """
    out = {"game_info": None, "articles": [], "reviews": [], "releases": []}
    if not isinstance(gamespot_wrapper, dict):
        return out

    games = gamespot_wrapper.get("games") or []
    # The fetcher stores each entry as {"game": {...}, "related": {"articles":[], "reviews":[], "releases":[]}}
    if isinstance(games, list) and games:
        # choose the first gamespot entry that has a game object
        entry = games[0]
        game_obj = entry.get("game") or {}
        out["game_info"] = game_obj

        related = entry.get("related") or {}
        # articles
        for a in related.get("articles") or []:
            out["articles"].append({
                "id": a.get("id") or a.get("guid") or None,
                "title": a.get("title") or a.get("deck") or None,
                "authors": a.get("authors") or a.get("author") or None,
                "deck": a.get("deck") or None,
                "body_html": a.get("body") or a.get("body_html") or None,
                "body_text": _strip_html(a.get("body") or a.get("body_html")) if a.get("body") else None,
                "site_detail_url": a.get("site_detail_url") or a.get("url"),
                "categories": a.get("categories") or [],
                "associations": a.get("associations") or [],
                "published_at": _parse_gamespot_date(a.get("publish_date") or a.get("published_at") or a.get("date")),
                "updated_at": _parse_gamespot_date(a.get("update_date") or a.get("updated_at"))
            })
        # reviews
        for r in related.get("reviews") or []:
            out["reviews"].append({
                "id": r.get("id") or r.get("guid") or None,
                "title": r.get("title") or r.get("deck") or None,
                "author": r.get("authors") or r.get("author") or None,
                "site_detail_url": r.get("site_detail_url") or r.get("url"),
                "review_text": _strip_html(r.get("body") or r.get("review") or r.get("body_html")) if (r.get("body") or r.get("review")) else None,
                "score": r.get("score") or r.get("rating") or None,
                "published_at": _parse_gamespot_date(r.get("publish_date") or r.get("published_at") or r.get("date"))
            })
        # releases
        for rel in related.get("releases") or []:
            out["releases"].append({
                "platform": _extract_name_from_item(rel.get("platform")) or rel.get("platform"),
                "region": rel.get("region") or None,
                "date": _parse_gamespot_date(rel.get("date") or rel.get("released_at") or rel.get("release_date")),
                "notes": rel.get("notes") or None
            })
    return out


def _strip_html(html: Optional[str]) -> Optional[str]:
    if not html or not isinstance(html, str):
        return None
    # extremely simple html stripper (conservative)
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _parse_gamespot_date(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    # try ISO parse, else try common formats, else return raw
    try:
        dt = datetime.fromisoformat(val)
        return dt.date().isoformat()
    except Exception:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d %b %Y"):
            try:
                return datetime.strptime(val, fmt).date().isoformat()
            except Exception:
                continue
    return val  # fallback: keep original


# ---------- Merge logic (three sources) ----------
def merge_three_sources(rawg: Dict[str, Any], igdb: Dict[str, Any], gamespot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge RAWG, IGDB, and GameSpot payloads into unified dict compatible with GameMerged.
    Uses same heuristics as your rough.py but extends with GameSpot and extra metadata.
    """
    # first reuse detection/unwrapping heuristics from rough.py style
    def _looks_like_rawg(rec: Any) -> bool:
        if not isinstance(rec, dict):
            return False
        if rec.get("description_raw") or rec.get("released") or rec.get("metacritic"):
            return True
        platforms = rec.get("platforms")
        if isinstance(platforms, list) and any(isinstance(p, dict) and p.get("platform") for p in platforms):
            return True
        return False

    def _looks_like_igdb(wrapper: Any) -> bool:
        if not isinstance(wrapper, dict):
            return False
        recs = wrapper.get("records")
        if isinstance(recs, list) and recs:
            first = recs[0]
            if isinstance(first, dict) and isinstance(first.get("clean"), list):
                return True
            if isinstance(first, dict) and (first.get("aggregated_rating") or first.get("first_release_date")):
                return True
        if isinstance(wrapper.get("genres"), list) and wrapper.get("genres") and all(isinstance(x, int) for x in wrapper.get("genres")):
            return True
        return False

    def _get_rawg_record(rawg_wrapper: Dict[str, Any]) -> Dict[str, Any]:
        if not rawg_wrapper:
            return {}
        if isinstance(rawg_wrapper, dict) and "records" in rawg_wrapper and isinstance(rawg_wrapper["records"], list) and rawg_wrapper["records"]:
            for candidate in rawg_wrapper["records"]:
                if _looks_like_rawg(candidate):
                    return candidate
            return rawg_wrapper["records"][0]
        if _looks_like_rawg(rawg_wrapper):
            return rawg_wrapper
        return rawg_wrapper

    def choose_igdb_clean_record(igdb_wrapper: Dict[str, Any], rawg_name: str = "", rawg_slug: str = "") -> Dict[str, Any]:
        if not igdb_wrapper:
            return {}
        clean_list = []
        if isinstance(igdb_wrapper, dict) and isinstance(igdb_wrapper.get("records"), list) and igdb_wrapper["records"]:
            first = igdb_wrapper["records"][0]
            if isinstance(first, dict) and isinstance(first.get("clean"), list):
                clean_list = first["clean"]
            else:
                clean_list = igdb_wrapper["records"]
        elif isinstance(igdb_wrapper, list):
            clean_list = igdb_wrapper

        if not clean_list:
            return {}

        rawg_name_l = (rawg_name or "").strip().lower()
        rawg_slug_l = (rawg_slug or "").strip().lower()

        for r in clean_list:
            if isinstance(r, dict) and r.get("name") and r["name"].strip().lower() == rawg_name_l:
                return r
        for r in clean_list:
            if isinstance(r, dict) and r.get("slug") and r["slug"].strip().lower() == rawg_slug_l:
                return r
        for r in clean_list:
            if isinstance(r, dict) and r.get("name") and rawg_name_l and rawg_name_l in r["name"].strip().lower():
                return r

        def score(r):
            if not isinstance(r, dict):
                return 0.0
            return float(r.get("aggregated_rating") or r.get("total_rating") or r.get("rating") or 0.0)
        best = max(clean_list, key=score)
        return best if isinstance(best, dict) else {}

    def _get_igdb_record(igdb_wrapper, rawg_rec):
        igdb_choice = choose_igdb_clean_record(igdb_wrapper, rawg_rec.get("name", ""), rawg_rec.get("slug", ""))
        if igdb_choice:
            return igdb_choice
        if isinstance(igdb_wrapper, dict) and isinstance(igdb_wrapper.get("records"), list) and igdb_wrapper["records"]:
            first = igdb_wrapper["records"][0]
            if isinstance(first, dict) and not isinstance(first.get("clean"), list):
                return first
        return igdb_wrapper

    merged: Dict[str, Any] = {}
    merged["source"] = {"rawg": rawg, "igdb": igdb, "gamespot": gamespot}

    rawg_candidate = rawg if _looks_like_rawg(rawg) else (rawg if rawg else {})
    igdb_candidate = igdb if _looks_like_igdb(igdb) else (igdb if igdb else {})

    rawg_rec = _get_rawg_record(rawg_candidate)
    igdb_rec = _get_igdb_record(igdb_candidate, rawg_rec)

    # GameSpot buckets
    gamespot_buckets = _collect_gamespot_buckets(gamespot)
    merged["gamespot"] = gamespot_buckets

    # IDs and unified id preference: RAWG > IGDB > GameSpot
    merged["rawg_id"] = rawg_rec.get("id")
    merged["igdb_id"] = igdb_rec.get("id")
    gp_id = None
    if isinstance(gamespot_buckets.get("game_info"), dict):
        gp_id = gamespot_buckets["game_info"].get("id") or gamespot_buckets["game_info"].get("guid") or gamespot_buckets["game_info"].get("site_detail_url")
    merged["gamespot_id"] = str(gp_id) if gp_id is not None else None

    if merged.get("rawg_id"):
        merged["unified_id"] = f"rawg:{merged['rawg_id']}"
    elif merged.get("igdb_id"):
        merged["unified_id"] = f"igdb:{merged['igdb_id']}"
    elif merged.get("gamespot_id"):
        merged["unified_id"] = f"gamespot:{merged['gamespot_id']}"
    else:
        merged["unified_id"] = None

    # Title & slug
    title = (
        (rawg_rec.get("name_original") if isinstance(rawg_rec, dict) else None)
        or rawg_rec.get("name")
        or igdb_rec.get("name")
        or igdb_rec.get("resolved_name")
        or (gamespot_buckets.get("game_info") or {}).get("name")
        or igdb_rec.get("slug")
        or rawg_rec.get("slug")
        or ""
    )
    merged["title"] = title.strip() if isinstance(title, str) else str(title)
    merged["slug"] = (igdb_rec.get("slug") or rawg_rec.get("slug") or (gamespot_buckets.get("game_info") or {}).get("site_detail_url") or None)

    # Description
    desc = None
    if isinstance(rawg_rec, dict):
        desc = rawg_rec.get("description_raw") or rawg_rec.get("description")
    if not desc:
        desc = igdb_rec.get("summary") or igdb_rec.get("storyline") or (gamespot_buckets.get("game_info") or {}).get("deck")
    merged["description"] = desc.strip() if isinstance(desc, str) else None

    # Release dates (collect multiple sources)
    release_dates = []
    # RAWG 'released' (single)
    rd = _iso_date_from_rawg(rawg_rec.get("released")) if isinstance(rawg_rec, dict) else None
    if rd:
        release_dates.append({"platform": None, "region": None, "date": rd.isoformat(), "source": "rawg"})
    # IGDB first_release_date
    id_rd = _date_from_unix(igdb_rec.get("first_release_date")) if isinstance(igdb_rec, dict) else None
    if id_rd:
        release_dates.append({"platform": None, "region": None, "date": id_rd.isoformat(), "source": "igdb"})
    # IGDB may have release_dates array
    for r in (igdb_rec.get("release_dates") or []):
        if isinstance(r, dict):
            d = None
            if r.get("date"):
                try:
                    d = datetime.fromisoformat(r.get("date")).date().isoformat()
                except Exception:
                    try:
                        d = datetime.utcfromtimestamp(int(r.get("date"))).date().isoformat()
                    except Exception:
                        d = None
            if not d and r.get("human"):
                d = r.get("human")
            release_dates.append({"platform": r.get("platform") or None, "region": r.get("region") or None, "date": d, "source": "igdb"})
    # GameSpot releases
    for r in (gamespot_buckets.get("releases") or []):
        release_dates.append({"platform": r.get("platform"), "region": r.get("region"), "date": r.get("date"), "source": "gamespot"})

    merged["release_dates"] = [rd for rd in release_dates if rd.get("date")]
    merged["release_date"] = merged["release_dates"][0]["date"] if merged["release_dates"] else None
    merged["release_year"] = int(merged["release_date"][:4]) if merged.get("release_date") else None

    # Platforms / genres / tags / themes / developers / publishers
    merged["platforms"] = _list_union_normalize(rawg_rec.get("platforms") if isinstance(rawg_rec, dict) else None,
                                                igdb_rec.get("platforms") if isinstance(igdb_rec, dict) else None,
                                                (gamespot_buckets.get("game_info") or {}).get("platforms"))
    merged["genres"] = _list_union_normalize(rawg_rec.get("genres") if isinstance(rawg_rec, dict) else None,
                                             igdb_rec.get("genres") if isinstance(igdb_rec, dict) else None,
                                             (gamespot_buckets.get("game_info") or {}).get("genres"))
    merged["tags"] = _list_union_normalize(rawg_rec.get("tags") if isinstance(rawg_rec, dict) else None,
                                           igdb_rec.get("tags") if isinstance(igdb_rec, dict) else None)
    merged["themes"] = _list_union_normalize(igdb_rec.get("themes") if isinstance(igdb_rec, dict) else None,
                                             igdb_rec.get("keywords") if isinstance(igdb_rec, dict) else None)

    # developers / publishers (IGDB involved_companies or RAWG developers/publishers)
    devs = []
    pubs = []
    # IGDB convention: involved_companies or developers/publishers arrays
    if isinstance(igdb_rec, dict):
        for key in ("involved_companies", "developers", "developers_names", "publishers", "publishers_names"):
            if igdb_rec.get(key):
                devs.extend(_list_union_normalize(igdb_rec.get("involved_companies") if key == "involved_companies" else igdb_rec.get(key)))
    # RAWG may provide 'developers' or 'publishers' arrays
    if isinstance(rawg_rec, dict):
        devs.extend(_list_union_normalize(rawg_rec.get("developers")))
        pubs.extend(_list_union_normalize(rawg_rec.get("publishers")))
    # fallback: game_info associations in GameSpot
    associations = (gamespot_buckets.get("game_info") or {}).get("associations") or []
    if associations:
        merged["tags"].extend([a.get("name") for a in associations if isinstance(a, dict) and a.get("name")])
    merged["developers"] = list(dict.fromkeys([d for d in devs if d]))  # preserve order unique
    merged["publishers"] = list(dict.fromkeys([p for p in pubs if p]))

    # Age ratings / esrb
    age_r = []
    if isinstance(igdb_rec, dict):
        for ar in (igdb_rec.get("age_ratings") or []):
            if isinstance(ar, dict):
                age_r.append(ar.get("rating") or ar.get("name"))
            elif isinstance(ar, (str, int)):
                age_r.append(str(ar))
    # RAWG esrb
    if isinstance(rawg_rec, dict) and rawg_rec.get("esrb"):
        maybe = rawg_rec.get("esrb")
        if isinstance(maybe, dict) and maybe.get("name"):
            age_r.append(maybe.get("name"))
    merged["age_ratings"] = [x for x in dict.fromkeys([a for a in age_r if a])]
    merged["esrb_rating"] = merged["age_ratings"][0] if merged["age_ratings"] else None

    # Ratings
    merged["ratings"] = {
        "rawg": rawg_rec.get("rating") if isinstance(rawg_rec, dict) else None,
        "igdb": (igdb_rec.get("aggregated_rating") or igdb_rec.get("total_rating") or igdb_rec.get("rating")) if isinstance(igdb_rec, dict) else None,
        "metacritic": rawg_rec.get("metacritic") if isinstance(rawg_rec, dict) else None,
        "rawg_detail": rawg_rec.get("ratings") if isinstance(rawg_rec, dict) else None,
        "igdb_detail": {k: igdb_rec.get(k) for k in ["aggregated_rating", "total_rating", "rating", "rating_count", "total_rating_count"] if isinstance(igdb_rec, dict) and igdb_rec.get(k) is not None},
        "normalized_0_100": None
    }
    # optionally compute normalized_0_100 if values present
    try:
        if merged["ratings"]["metacritic"]:
            merged["ratings"]["normalized_0_100"] = int(merged["ratings"]["metacritic"])
        elif merged["ratings"]["igdb"]:
            # IGDB ratings often 0-100 float
            merged["ratings"]["normalized_0_100"] = int(round(float(merged["ratings"]["igdb"])))
        elif merged["ratings"]["rawg"]:
            # RAWG rating often 0-5 float; scale to 0-100
            merged["ratings"]["normalized_0_100"] = int(round(float(merged["ratings"]["rawg"]) * 20))
    except Exception:
        merged["ratings"]["normalized_0_100"] = None

    # URLs / websites / stores
    urls = set()
    websites = set()
    stores = set()
    if isinstance(rawg_rec, dict):
        if rawg_rec.get("website"):
            websites.add(rawg_rec.get("website"))
        if rawg_rec.get("metacritic_url"):
            urls.add(rawg_rec.get("metacritic_url"))
        for s in (rawg_rec.get("stores") or []):
            if isinstance(s, dict) and s.get("url"):
                stores.add(s.get("url"))
            elif isinstance(s, str):
                stores.add(s)
    if isinstance(igdb_rec, dict):
        if igdb_rec.get("url"):
            urls.add(igdb_rec.get("url"))
        for w in (igdb_rec.get("websites") or []):
            if isinstance(w, dict) and w.get("url"):
                websites.add(w.get("url"))
            elif isinstance(w, str):
                websites.add(w)
    # GameSpot links: site_detail_url in articles/reviews and game_info
    gp = gamespot_buckets.get("game_info") or {}
    if gp.get("site_detail_url"):
        urls.add(gp.get("site_detail_url"))
    for a in (gamespot_buckets.get("articles") or []):
        if a.get("site_detail_url"):
            urls.add(a.get("site_detail_url"))
    for r in (gamespot_buckets.get("reviews") or []):
        if r.get("site_detail_url"):
            urls.add(r.get("site_detail_url"))

    merged["urls"] = [u for u in list(urls) if u]
    merged["websites"] = [w for w in list(websites) if w]
    merged["stores"] = [s for s in list(stores) if s]

    # gamespot bucket already captured
    # build documents from gamespot articles & reviews (flatten for RAG)
    documents = []
    for a in (gamespot_buckets.get("articles") or []):
        doc = {
            "id": a.get("id"),
            "source": "gamespot:article",
            "title": a.get("title"),
            "content": a.get("body_text") or a.get("deck") or "",
            "excerpt": a.get("deck"),
            "created_at": a.get("published_at"),
            "meta": {"site_detail_url": a.get("site_detail_url"), "categories": a.get("categories")}
        }
        documents.append(doc)
    for r in (gamespot_buckets.get("reviews") or []):
        doc = {
            "id": r.get("id"),
            "source": "gamespot:review",
            "title": r.get("title"),
            "content": r.get("review_text") or "",
            "excerpt": None,
            "created_at": r.get("published_at"),
            "meta": {"site_detail_url": r.get("site_detail_url"), "score": r.get("score")}
        }
        documents.append(doc)
    merged["documents"] = documents

    # merged_from decisions (which source provided what)
    merged["merged_from"] = {
        "title_from": "rawg" if (isinstance(rawg_rec, dict) and (rawg_rec.get("name") or rawg_rec.get("name_original"))) else ("igdb" if (isinstance(igdb_rec, dict) and igdb_rec.get("name")) else "gamespot"),
        "description_from": "rawg" if (isinstance(rawg_rec, dict) and (rawg_rec.get("description_raw") or rawg_rec.get("description"))) else ("igdb" if (isinstance(igdb_rec, dict) and (igdb_rec.get("summary") or igdb_rec.get("storyline"))) else "gamespot"),
        "release_from": "rawg" if rd else ("igdb" if id_rd else "gamespot")
    }

    return merged


# ---------- CLI / convenience runner ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge RAWG + IGDB + GameSpot JSON into unified schema")
    parser.add_argument("--rawg", default="far_cry_5_rawg.json", help="Path to RAWG JSON")
    parser.add_argument("--igdb", default="far_cry_5_igdb.json", help="Path to IGDB JSON")
    parser.add_argument("--gamespot", default="far_cry_5_gamespot_full_textual.json", help="Path to GameSpot full textual JSON")
    parser.add_argument("--out", default="merged_three_sources.json", help="Output filename")
    args = parser.parse_args()

    try:
        rawg = load_json(args.rawg)
        igdb = load_json(args.igdb)
        gamespot = load_json(args.gamespot)
    except FileNotFoundError as e:
        print("File missing:", e)
        raise SystemExit(1)

    merged = merge_three_sources(rawg, igdb, gamespot)

    # Validate with Pydantic (coerce types) and save
    gm = GameMerged.parse_obj({
        "unified_id": merged.get("unified_id"),
        "title": merged.get("title"),
        "slug": merged.get("slug"),
        "description": merged.get("description"),
        "release_date": merged.get("release_date"),
        "release_year": merged.get("release_year"),
        "release_dates": merged.get("release_dates"),
        "rawg_id": merged.get("rawg_id"),
        "igdb_id": merged.get("igdb_id"),
        "gamespot_id": merged.get("gamespot_id"),
        "platforms": merged.get("platforms"),
        "genres": merged.get("genres"),
        "tags": merged.get("tags"),
        "themes": merged.get("themes"),
        "developers": merged.get("developers"),
        "publishers": merged.get("publishers"),
        "age_ratings": merged.get("age_ratings"),
        "esrb_rating": merged.get("esrb_rating"),
        "ratings": merged.get("ratings"),
        "urls": merged.get("urls"),
        "websites": merged.get("websites"),
        "stores": merged.get("stores"),
        "gamespot": merged.get("gamespot"),
        "documents": merged.get("documents"),
        "source": merged.get("source"),
        "merged_from": merged.get("merged_from"),
    })

    save_json(gm.dict(), args.out)
    print("Merged file saved to:", args.out)
