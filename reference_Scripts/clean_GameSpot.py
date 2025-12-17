#!/usr/bin/env python3
"""
GameSpot Strict Schema Transformer (FIXED)
-----------------------------------------
Correctly maps the GameSpot fetcher structure:

root
 └── games[0]
      ├── game
      └── related
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


# ----------------------------
# Helpers
# ----------------------------

def iso_date(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(val[:len(fmt)], fmt).date().isoformat()
        except Exception:
            continue
    return None


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def avg(nums: List[float]) -> Optional[float]:
    nums = [n for n in nums if isinstance(n, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


# ----------------------------
# Transformation Logic
# ----------------------------

def transform(source: Dict[str, Any]) -> Dict[str, Any]:
    """
    STRICT mapping according to GAmeSpot_schema
    """

    # ---- STEP 1: Navigate correctly ----
    games = safe_list(source.get("games"))
    first = games[0] if games else {}

    game = first.get("game", {})
    related = first.get("related", {})

    reviews_src = safe_list(related.get("reviews"))
    articles_src = safe_list(related.get("articles"))

    review_scores = [r.get("score") for r in reviews_src]

    # ---- STEP 2: Build STRICT target object ----
    return {
        "metadata": {
            "id": game.get("id"),
            "name": game.get("name"),
            "slug": game.get("slug"),
            "release_date": iso_date(
                game.get("original_release_date") or game.get("release_date")
            ),
        },
        "summary": {
            "deck": game.get("deck"),
            "description": game.get("description"),
        },
        "reviews": {
            "average_score": avg(review_scores),
            "items": [
                {
                    "title": r.get("title"),
                    "score": r.get("score"),
                    "date": iso_date(r.get("publish_date")),
                    "body": r.get("review_text") or r.get("body"),
                }
                for r in reviews_src
            ],
        },
        "articles": [
            {
                "title": a.get("title"),
                "date": iso_date(a.get("publish_date")),
                "deck": a.get("deck"),
                "body": a.get("body"),
            }
            for a in articles_src
        ],
    }


# ----------------------------
# Entrypoint
# ----------------------------

def main():
    with open("assassins_creed_valhalla_gamespot_full_textual.json", "r", encoding="utf-8") as f:
        source_data = json.load(f)

    cleaned = transform(source_data)

    with open("cleaned_data_output.json", "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print("[OK] cleaned_data_output.json generated successfully")


if __name__ == "__main__":
    main()