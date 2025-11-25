# validate_merged.py
import json, sys, os
from datetime import datetime

PATH = "merged_three_sources.json"   # adjust path if needed

REQUIRED_FIELDS = {
    "slug": str,
    "unified_game_id": (str, type(None)),
    "title": str,
    "description": (str, type(None)),
    "release_date": (str, type(None)),
    "release_year": (int, type(None)),
    "genres": (list, type(None)),
    "platforms": (list, type(None)),
    "developers": (list, type(None)),
    "publishers": (list, type(None)),
    "rating": (int, float, type(None)),
    "metacritic": (int, float, type(None)),
    "esrb_rating": (str, type(None)),
    "playtime": (int, float, type(None)),
    "site_detail_url": (str, type(None)),
    "articles": (list, type(None)),
    "articles_count": (int, type(None)),
    "reviews": (list, type(None)),
    "source": dict,
    "text": (str, type(None)),
    # ratings.rawg_detail special case: accept dict or list
    "ratings.rawg_detail": ("dict_or_list",)
}

def check_type(val, expected):
    if expected == ("dict_or_list",):
        return isinstance(val, (dict, list))
    if isinstance(expected, tuple):
        return any(isinstance(val, t) for t in expected)
    return isinstance(val, expected)

def get_nested(obj, path):
    parts = path.split(".")
    cur = obj
    for p in parts:
        if cur is None:
            return None
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def main():
    if not os.path.exists(PATH):
        print("File not found:", PATH); sys.exit(2)
    with open(PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If top is list, check first merged game; if dict, try to find a canonical 'game' or first object
    candidates = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        # if there is a top-level 'data' or 'sources' or 'game' key try those
        if "data" in data and isinstance(data["data"], list):
            candidates = data["data"]
        elif "game" in data:
            candidates = [data]
        else:
            # fallback: use dict itself as single merged object
            candidates = [data]

    # validate first candidate only (adjust as needed)
    obj = candidates[0] if candidates else {}
    print("Validating object sample (top-level keys):", list(obj.keys())[:40])

    results = []
    for field, expected in REQUIRED_FIELDS.items():
        if "." in field:
            # nested path
            val = get_nested(obj, field)
        else:
            val = obj.get(field)
        ok = check_type(val, expected) if val is not None else expected is not None and check_type(val, expected)
        results.append((field, ok, type(val).__name__, val if (isinstance(val, (str, int, float, list, dict))) else "VALUE_PRESENT"))

    passed = sum(1 for r in results if r[1])
    total = len(results)
    print("\nField validation results:")
    for f, ok, tname, val in results:
        print(f" - {f:30s} : {'PASS' if ok else 'FAIL':4s} (type:{tname})")

    print("\nSCORE: {}/{} = {:.2f}%".format(passed, total, passed/total*100))
    # Print a few suspect fields for debugging
    suspect = [r for r in results if not r[1]]
    if suspect:
        print("\nSuspect / failing fields (showing sample values):")
        for f, ok, tname, val in suspect:
            print(" *", f, "->", tname, "sample:", str(val)[:300])

if __name__ == '__main__':
    main()
