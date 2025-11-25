#!/usr/bin/env python3
"""
merge_gamespot_no_schema.py

Merge GameSpot textual JSON into an existing merged IGDB+RAWG JSON WITHOUT using the schema Excel.

Input files (paths hard-coded to the files you provided):
 - /mnt/data/merged_far_cry_5.json
 - /mnt/data/far_cry_5_gamespot_full_textual.json

Outputs:
 - /mnt/data/merged_far_cry_5_with_gamespot.json
 - /mnt/data/merged_far_cry_5_with_gamespot_validation.json
"""

import json
import os
import re
import hashlib
from difflib import SequenceMatcher
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, Set

# === Config: file paths (from your uploads) ===
BASE_PATH = "merged_far_cry_5.json"
GAMESPOT_PATH = "far_cry_5_gamespot_full_textual.json"
OUT_PATH = "merged_far_cry_5_with_gamespot.json"
VALIDATION_OUT = "merged_far_cry_5_with_gamespot_validation.json"

# === Heuristics / thresholds ===
NAME_MATCH_THRESHOLD = 0.80  # >= this we consider gamespot entry matches the base record
FUZZY_FIELD_MATCH_THRESHOLD = 0.7  # used only for optional fuzzy mapping if needed

# === Utilities ===

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.strip()
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def dedup_preserve_order(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x is None:
            continue
        key = x.strip().lower() if isinstance(x, str) else x
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

def stable_union_lists(a: Optional[List[Any]], b: Optional[List[Any]]) -> List[Any]:
    if not a and not b:
        return []
    a_list = list(a) if a else []
    b_list = list(b) if b else []
    # convert non-strings to string for dedup key but preserve original item
    out: List[Any] = []
    seen: Set[str] = set()
    for item in (a_list + b_list):
        if item is None:
            continue
        key = (str(item)).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out

def compute_unified_id(obj: Dict[str, Any]) -> str:
    if obj.get("unified_id"):
        return obj["unified_id"]
    # prefer rawg_id or igdb_id if present
    if obj.get("rawg_id"):
        return f"rawg:{obj['rawg_id']}"
    if obj.get("igdb_id"):
        return f"igdb:{obj['igdb_id']}"
    # otherwise compute sha1 over title/ids
    base = json.dumps({
        "title": obj.get("title"),
        "igdb_id": obj.get("igdb_id"),
        "rawg_id": obj.get("rawg_id")
    }, sort_keys=True).encode("utf-8")
    return "unified:" + hashlib.sha1(base).hexdigest()[:12]

# === GameSpot extraction ===

def extract_gamespot_entries(gamespot_data: Any) -> List[Dict[str, Any]]:
    """
    Extract a list of candidate textual GameSpot entries from the uploaded file.
    Heuristics: if top-level has 'games' list, use that; else if it's a single dict containing name/title use it;
    else scan lists inside top-level and take dict-like entries.
    """
    entries = []
    if isinstance(gamespot_data, dict):
        if "games" in gamespot_data and isinstance(gamespot_data["games"], list):
            for item in gamespot_data["games"]:
                # if wrapper with 'game' key, prefer that
                if isinstance(item, dict) and "game" in item and isinstance(item["game"], dict):
                    # merge game + related textual parts to one doc
                    merged = dict(item["game"])
                    if "related" in item and isinstance(item["related"], dict):
                        merged["related"] = item["related"]
                    entries.append(merged)
                elif isinstance(item, dict):
                    entries.append(item)
        else:
            # single game object or other structure
            if "name" in gamespot_data or "title" in gamespot_data or "id" in gamespot_data:
                entries.append(gamespot_data)
            else:
                # scan for lists with dict entries
                for v in gamespot_data.values():
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict) and ("name" in it or "title" in it or "id" in it):
                                entries.append(it)
    elif isinstance(gamespot_data, list):
        for it in gamespot_data:
            if isinstance(it, dict):
                entries.append(it)
    return entries

# === Merge logic ===

def merge_ratings(base_ratings: Optional[Dict[str, Any]], incoming_ratings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base_ratings or {})
    inc = incoming_ratings or {}
    for k, v in inc.items():
        if k not in out or out[k] in (None, "", 0):
            out[k] = v
        else:
            # if both dicts merge shallow
            if isinstance(out[k], dict) and isinstance(v, dict):
                merged = dict(out[k])
                for subk, subv in v.items():
                    if merged.get(subk) in (None, "", 0):
                        merged[subk] = subv
                out[k] = merged
            # else keep existing
    return out

def merge_gamespot_into_base(base: Dict[str, Any], gs_entry: Dict[str, Any], prefer_incoming_fields: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Merge a single GameSpot entry into the base merged object.
    Returns (updated_base, changes_summary)
    - prefer_incoming_fields: if True, overwrite base scalars when incoming present.
      Default False: conservative (only fill missing fields).
    """
    changes = {"set": {}, "list_union": {}, "ratings": {}, "ids_set": {}}

    # ensure source.gamespot exists and store raw payload (append)
    base.setdefault("source", {})
    base["source"].setdefault("gamespot", [])
    base["source"]["gamespot"].append(gs_entry)

    # title/name mapping
    base_title = base.get("title") or base.get("name") or ""
    gs_name = gs_entry.get("name") or gs_entry.get("title") or ""
    # Scalars to consider merging; conservative: don't overwrite unless prefer_incoming_fields True
    scalar_fields = ["title", "description", "slug", "summary", "story", "release_date", "developer", "publisher"]
    for sf in scalar_fields:
        inc_val = gs_entry.get(sf)
        # If incoming has value and base missing -> set
        if inc_val not in (None, "", [], {}):
            if base.get(sf) in (None, "", [], {}):
                base[sf] = inc_val
                changes["set"][sf] = {"from": None, "to": inc_val}
            else:
                # optional overwrite if user wants
                if prefer_incoming_fields and base.get(sf) != inc_val:
                    old = base.get(sf)
                    base[sf] = inc_val
                    changes["set"][sf] = {"from": old, "to": inc_val}

    # lists: platforms, genres, tags, urls
    for list_key in ("platforms", "genres", "tags", "urls"):
        base_list = base.get(list_key) or []
        incoming_list = gs_entry.get(list_key) or []
        merged_list = stable_union_lists(base_list, incoming_list)
        if merged_list != base_list:
            base[list_key] = merged_list
            changes["list_union"][list_key] = {"from_count": len(base_list), "to_count": len(merged_list)}

    # ratings: preserve per-source structure; put under ratings.gamespot
    base.setdefault("ratings", {})
    gs_ratings = gs_entry.get("ratings") or gs_entry.get("score") or gs_entry.get("rating") or {}
    # If gamespot ratings are simple numeric, wrap under 'gamespot' key
    if isinstance(gs_ratings, (int, float, str)):
        gs_ratings = {"score": gs_ratings}
    base["ratings"] = merge_ratings(base.get("ratings"), {"gamespot": gs_ratings})
    changes["ratings"]["gamespot"] = gs_ratings

    # ids & urls
    # common possible keys: 'id', 'gamespot_id', 'site_detail_url', 'url'
    if gs_entry.get("id") and not base.get("gamespot_id"):
        base["gamespot_id"] = gs_entry.get("id")
        changes["ids_set"]["gamespot_id"] = gs_entry.get("id")
    if gs_entry.get("url") and gs_entry.get("url") not in (None, ""):
        base.setdefault("urls", [])
        if gs_entry["url"] not in base["urls"]:
            base["urls"].append(gs_entry["url"])
            changes["ids_set"]["gamespot_url"] = gs_entry["url"]
    # also check nested 'site_detail_url' or related link arrays
    if gs_entry.get("site_detail_url"):
        base.setdefault("urls", [])
        if gs_entry["site_detail_url"] not in base["urls"]:
            base["urls"].append(gs_entry["site_detail_url"])
            changes["ids_set"]["gamespot_site_detail_url"] = gs_entry["site_detail_url"]

    # merged_from provenance
    base.setdefault("merged_from", {})
    # mark which fields came from gamespot if we set them
    for k in list(changes["set"].keys()) + list(changes["list_union"].keys()) + list(changes["ids_set"].keys()):
        base["merged_from"][f"{k}_from"] = "gamespot"

    # ensure unified_id exists
    base["unified_id"] = compute_unified_id(base)

    return base, changes

# === Main orchestration ===

def main():
    if not os.path.exists(BASE_PATH):
        raise SystemExit(f"Base merged file not found: {BASE_PATH}")
    if not os.path.exists(GAMESPOT_PATH):
        raise SystemExit(f"GameSpot file not found: {GAMESPOT_PATH}")

    base = load_json(BASE_PATH)
    gamespot_data = load_json(GAMESPOT_PATH)
    print(f"[INFO] Loaded base merged JSON from: {BASE_PATH}")
    print(f"[INFO] Loaded GameSpot JSON from: {GAMESPOT_PATH}")

    gs_entries = extract_gamespot_entries(gamespot_data)
    print(f"[INFO] Extracted {len(gs_entries)} GameSpot textual entries to consider merging.")

    # We'll attempt to match each gamespot entry to the base record by name/title similarity (since base is ONE record)
    base_name = normalize_text(base.get("title") or base.get("name") or "")
    merged_count = 0
    attached_reports = []
    unmatched_entries = []

    for gs in gs_entries:
        gs_name = normalize_text(gs.get("name") or gs.get("title") or "")
        score = similar(base_name, gs_name) if base_name and gs_name else 0.0
        if score >= NAME_MATCH_THRESHOLD:
            # merge into base
            base, changes = merge_gamespot_into_base(base, gs, prefer_incoming_fields=False)
            merged_count += 1
            attached_reports.append({"gamespot_name": gs.get("name") or gs.get("title"), "score": score, "changes": changes})
        else:
            # didn't match; save for manual review
            unmatched_entries.append({"gamespot_name": gs.get("name") or gs.get("title"), "score": score, "raw": gs})

    # if no base_name (rare), attempt to merge the best candidate (highest similarity) if it looks close to 0.6+
    if not base_name and gs_entries:
        # pick first entry as source of truth
        gs = gs_entries[0]
        base, changes = merge_gamespot_into_base(base, gs, prefer_incoming_fields=False)
        attached_reports.append({"gamespot_name": gs.get("name") or gs.get("title"), "score": None, "changes": changes})
        merged_count += 1

    # store unmatched entries under a safe key for inspection
    if unmatched_entries:
        base.setdefault("gamespot_unmatched", [])
        base["gamespot_unmatched"].extend(unmatched_entries)

    # finalize unified_id and provenance timestamp
    base["unified_id"] = compute_unified_id(base)
    base.setdefault("meta", {})
    base["meta"]["last_merged_from_gamespot_at"] = datetime.utcnow().isoformat() + "Z"

    # compute simple validation score over top-level keys present in base (before/after notion removed;
    # since no schema, we'll use the base's own keys as the expected set)
    top_fields = list(base.keys())
    present_count = sum(1 for k in top_fields if base.get(k) not in (None, "", [], {}))
    total = len(top_fields)
    validation = {
        "total_top_level_fields": total,
        "present_nonempty_top_level_fields": present_count,
        "score": f"{present_count}/{total}",
        "merged_entries_count": merged_count,
        "attached_reports_sample": attached_reports[:10],
        "unmatched_entries_count": len(unmatched_entries),
        "unmatched_sample": unmatched_entries[:5]
    }

    # Save outputs
    save_json(base, OUT_PATH)
    save_json(validation, VALIDATION_OUT)

    print(f"[OK] Merged file written to: {OUT_PATH}")
    print(f"[OK] Validation report written to: {VALIDATION_OUT}")
    print(f"[SUMMARY] GamesSpot entries merged into base: {merged_count}")
    print(f"[SUMMARY] GamesSpot entries left unmatched: {len(unmatched_entries)}")
    if attached_reports:
        print("\n[SAMPLE MERGE REPORT]")
        for r in attached_reports[:5]:
            print(json.dumps(r, indent=2, ensure_ascii=False))
    if unmatched_entries:
        print("\n[SAMPLE UNMATCHED]")
        for u in unmatched_entries[:5]:
            print(json.dumps(u, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
