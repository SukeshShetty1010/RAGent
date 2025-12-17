"""
RAWG JSON Cleaning & Restructuring Script
=======================================

Implements the cleaning_strategy, target_schema, and edge_case_handling
defined in Rawg_schema.txt and applies them to a RAWG raw JSON object.

Author: Senior Python Data Engineer (ETL)
"""

import json
import re
from datetime import datetime
from html import unescape
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------
# Helper Function: Text Sanitization
# Implements Strategy Rule #1: Text Sanitization
# ---------------------------------------------------------
def strip_html(text: Optional[str]) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


# ---------------------------------------------------------
# Helper Function: Deduplicate strings (case-insensitive)
# Implements Strategy Rule #6: Taxonomy Normalization
# ---------------------------------------------------------
def dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for v in values:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            result.append(v)
    return result


# ---------------------------------------------------------
# Helper Function: Build Platform Family Lookup
# Uses parent_platforms as fallback only (Edge Case Handling)
# ---------------------------------------------------------
def build_platform_family_lookup(parent_platforms: List[Dict[str, Any]]) -> Dict[str, str]:
    lookup = {}
    for p in parent_platforms or []:
        platform = p.get("platform") or {}
        name = platform.get("name")
        if name:
            lookup[name] = name
    return lookup


# ---------------------------------------------------------
# Main Transformation Function
# ---------------------------------------------------------
def transform_rawg(raw: Dict[str, Any]) -> Dict[str, Any]:
    # -----------------------------------------------------
    # Core Identifiers & Metadata
    # Implements Strategy Rule #2
    # -----------------------------------------------------
    game_id = raw.get("id")
    title = raw.get("name")
    description = strip_html(raw.get("description") or raw.get("description_raw"))

    release_date = raw.get("released")
    release_year = None
    if release_date:
        try:
            release_year = int(release_date[:4])
        except Exception:
            release_year = None

    # -----------------------------------------------------
    # Platform Modeling
    # Implements Strategy Rule #3 & Edge Case Handling
    # -----------------------------------------------------
    parent_lookup = build_platform_family_lookup(raw.get("parent_platforms") or [])
    platforms_out = []

    platforms = raw.get("platforms") or []
    if not platforms:
        # fallback to parent_platforms if platforms missing
        for pname in parent_lookup.keys():
            platforms_out.append({
                "platform_name": pname,
                "platform_family": pname,
                "release_date": None,
                "requirements_minimum": None,
                "requirements_recommended": None
            })
    else:
        for p in platforms:
            platform_info = p.get("platform") or {}
            requirements = p.get("requirements") or {}

            platform_name = platform_info.get("name")
            platforms_out.append({
                "platform_name": platform_name,
                "platform_family": parent_lookup.get(platform_name),
                "release_date": p.get("released_at"),
                "requirements_minimum": requirements.get("minimum"),
                "requirements_recommended": requirements.get("recommended"),
            })

    # -----------------------------------------------------
    # Taxonomy Normalization
    # Implements Strategy Rule #6
    # -----------------------------------------------------
    genres = dedupe_strings([
        g.get("name") for g in (raw.get("genres") or []) if g.get("name")
    ])

    developers = dedupe_strings([
        d.get("name") for d in (raw.get("developers") or []) if d.get("name")
    ])

    publishers = dedupe_strings([
        p.get("name") for p in (raw.get("publishers") or []) if p.get("name")
    ])

    # -----------------------------------------------------
    # Ratings Strategy
    # Implements Strategy Rule #5
    # -----------------------------------------------------
    ratings_out = {
        "average_rating": raw.get("rating"),
        "rating_top": raw.get("rating_top"),
        "metacritic": raw.get("metacritic"),
        "distribution": [],
        "metacritic_by_platform": []
    }

    for r in raw.get("ratings") or []:
        ratings_out["distribution"].append({
            "label": r.get("title"),
            "percent": float(r.get("percent")) if r.get("percent") is not None else None,
            "count": int(r.get("count")) if r.get("count") is not None else None
        })

    for mp in raw.get("metacritic_platforms") or []:
        plat = mp.get("platform") or {}
        ratings_out["metacritic_by_platform"].append({
            "platform": plat.get("name"),
            "score": mp.get("metascore")
        })

    # -----------------------------------------------------
    # Tags Filtering
    # Implements Strategy Rule #7
    # -----------------------------------------------------
    tags = [
        t.get("name")
        for t in (raw.get("tags") or [])
        if t.get("language") == "eng" and t.get("name")
    ]

    # -----------------------------------------------------
    # Engagement Metrics
    # Implements Strategy Rule #8
    # -----------------------------------------------------
    engagement = {
        "added_by_status": raw.get("added_by_status") or {},
        "reactions": raw.get("reactions") or {}
    }

    # -----------------------------------------------------
    # Age Rating
    # Implements Strategy Rule #9
    # -----------------------------------------------------
    age_rating = {
        "esrb": raw.get("esrb_rating")
    }

    # -----------------------------------------------------
    # Source Metadata
    # -----------------------------------------------------
    source = {
        "rawg_url": raw.get("website"),
        "last_updated": raw.get("updated")
    }

    # -----------------------------------------------------
    # Final Output (Target Schema)
    # -----------------------------------------------------
    return {
        "game_id": game_id,
        "title": title,
        "description": description,
        "release_date": release_date,
        "release_year": release_year,
        "platforms": platforms_out,
        "genres": genres,
        "developers": developers,
        "publishers": publishers,
        "ratings": ratings_out,
        "tags": tags,
        "age_rating": age_rating,
        "engagement": engagement,
        "source": source
    }


# ---------------------------------------------------------
# Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    input_file = "rawg_only_assassins_creed_valhalla.json"
    output_file = "cleaned_rawg_data.json"

    try:
        # 1. Read the raw file
        with open(input_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        print(f"Read data from: {input_file}")

        # 2. Transform the data
        cleaned = transform_rawg(raw_data)

        # 3. Save to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

        print(f"Success! Cleaned data saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")