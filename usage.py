#!/usr/bin/env python3
"""
validation_complete_merge.py

Validate the final merged JSON produced by merge_all_three.py.
Produces a structured validation report for debugging merge quality.

Usage:
    python validation_complete_merge.py --file far_cry_5_complete_merged.json --out validation_report.json
"""

import json
import argparse
import re
from datetime import datetime

# ------------ Helpers ------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    import os
    os.replace(tmp, path)

def is_valid_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    return bool(re.match(r"^https?://", url.strip()))

def count_nonempty(v):
    if v is None:
        return 0
    if isinstance(v, dict) and len(v) != 0:
        return 1
    if isinstance(v, list) and len(v) != 0:
        return 1
    if isinstance(v, str) and v.strip() != "":
        return 1
    if isinstance(v, (int, float)) and v != 0:
        return 1
    return 0

# ------------ Validation Logic ------------

def validate(merged: dict) -> dict:
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "title": merged.get("title"),
        "unified_id": merged.get("unified_id"),
        "field_completeness": {},
        "ratings_summary": {},
        "url_analysis": {},
        "source_summary": {},
        "gamespot_matching": {},
        "missing_fields": [],
        "raw_source_sizes": {},
        "structure_ok": True,
    }

    # === Field completeness ===
    key_fields = [
        "title", "description", "release_date", "platforms",
        "genres", "tags", "urls", "ratings", "slug"
    ]

    completeness = {}
    missing = []
    for k in key_fields:
        val = merged.get(k)
        filled = count_nonempty(val)
        completeness[k] = bool(filled)
        if not filled:
            missing.append(k)

    report["field_completeness"] = completeness
    report["missing_fields"] = missing

    # === Ratings summary ===
    ratings = merged.get("ratings", {})
    rating_report = {
        "rawg_rating_present": "rawg" in ratings and ratings.get("rawg") not in (None, 0, "", []),
        "igdb_rating_present": "igdb" in ratings and ratings.get("igdb") not in (None, 0, "", []),
        "metacritic_present": "metacritic" in ratings and ratings.get("metacritic") not in (None, 0, "", []),
        "gamespot_rating_present": "gamespot" in ratings,
        "raw_ratings_object": ratings
    }
    report["ratings_summary"] = rating_report

    # === URL analysis ===
    urls = merged.get("urls", [])
    valid_urls = []
    invalid_urls = []

    if isinstance(urls, list):
        for u in urls:
            if is_valid_url(u):
                valid_urls.append(u)
            else:
                invalid_urls.append(u)

    report["url_analysis"] = {
        "total": len(urls),
        "valid": len(valid_urls),
        "invalid": len(invalid_urls),
        "invalid_list": invalid_urls,
    }

    # === Source summary ===
    src = merged.get("source", {})
    src_summary = {
        "has_rawg": "rawg" in src and src["rawg"] is not None,
        "has_igdb": "igdb" in src and src["igdb"] is not None,
        "gamespot_entries_count": len(src.get("gamespot", [])) if isinstance(src.get("gamespot"), list) else 0
    }
    report["source_summary"] = src_summary

    # === GameSpot matching ===
    gs_unmatched = merged.get("gamespot_unmatched", [])
    match_report = {
        "matched_count": src_summary["gamespot_entries_count"],
        "unmatched_count": len(gs_unmatched),
        "sample_unmatched": gs_unmatched[:3]
    }
    report["gamespot_matching"] = match_report

    # === Raw source sizes ===
    for key in ("rawg", "igdb"):
        data = src.get(key)
        if isinstance(data, dict):
            report["raw_source_sizes"][key] = len(data)
        elif isinstance(data, list):
            report["raw_source_sizes"][key] = len(data)
        else:
            report["raw_source_sizes"][key] = 0

    # === Structural integrity ===
    if "title" not in merged or merged.get("title") is None:
        report["structure_ok"] = False

    if "ratings" not in merged or not isinstance(merged.get("ratings"), dict):
        report["structure_ok"] = False

    return report


# ------------ Main ------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to merged JSON (e.g., far_cry_5_complete_merged.json)")
    parser.add_argument("--out", default="validation_report.json", help="Output JSON path")
    args = parser.parse_args()

    merged = load_json(args.file)
    print(f"[INFO] Loaded merged JSON: {args.file}")

    report = validate(merged)

    save_json(report, args.out)
    print(f"[OK] Validation report saved to: {args.out}")

if __name__ == "__main__":
    main()
