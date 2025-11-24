import os
import sys
import json
import time
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
from dotenv import load_dotenv

# ---------------------------
# Load environment
# ---------------------------

load_dotenv()

# RAWG API key (for name correction)
RAWG_API_KEY = os.getenv("RAWG_API_KEY")

# GameSpot base
BASE_URL = "https://www.gamespot.com/api"
USER_AGENT = "RAG-ent-EldenRingFetcher/1.0 (contact: you@example.com)"  # set something non-generic

# Output file; we'll override the name based on the resolved title
DEFAULT_OUTPUT_FILE = "gamespot_full_textual.json"

# Visual-related key substrings to strip everywhere (case-insensitive)
VISUAL_KEY_SUBSTRINGS = [
    "image",
    "images",
    "video",
    "videos",
    "screenshot",
    "thumbnail",
    "thumbnails",
    "poster",
    "banner",
    "icon",
    "avatar",
    "logo",
]


# ---------------------------
# RAWG helper – find correct game name
# ---------------------------

def get_correct_game_name(query: str) -> str | None:
    """
    Searches RAWG for the closest matching game name and returns the official title.

    Args:
        query (str): Name of the game input by user.

    Returns:
        str | None: Returns the correct game title if found, otherwise None.
    """
    if not RAWG_API_KEY:
        raise ValueError("Missing RAWG_API_KEY in .env file!")

    url = "https://api.rawg.io/api/games"
    params = {
        "key": RAWG_API_KEY,
        "search": query,
        "page_size": 1,  # Return the closest match only
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            best_match = data["results"][0]
            return best_match.get("name")

        return None

    except requests.RequestException as e:
        print(f"Error searching RAWG: {e}")
        return None


def resolve_game_name(raw_query: str) -> str:
    """
    Use RAWG to normalize the game name if possible.
    Falls back to raw_query if RAWG can't help.
    """
    try:
        corrected = get_correct_game_name(raw_query)
    except ValueError as e:
        # Missing RAWG_API_KEY – warn and just use the raw name
        print(f"[WARN] {e}. Using '{raw_query}' as-is.")
        return raw_query

    if corrected:
        print(f"✔ Using RAWG-corrected name: '{corrected}' (from '{raw_query}')")
        return corrected

    print(f"[INFO] RAWG returned no better match; using '{raw_query}' as-is.")
    return raw_query


# ---------------------------
# Visual-stripping helpers
# ---------------------------

def is_visual_key(key: str) -> bool:
    """Return True if the key name clearly refers to visual content."""
    k = key.lower()
    return any(sub in k for sub in VISUAL_KEY_SUBSTRINGS)


def strip_visual_fields(obj):
    """
    Recursively remove any dict keys whose name looks visual-related
    (image/video/etc). Works for nested dicts/lists.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if is_visual_key(k):
                # Drop visual fields entirely
                continue
            cleaned[k] = strip_visual_fields(v)
        return cleaned
    elif isinstance(obj, list):
        return [strip_visual_fields(x) for x in obj]
    else:
        return obj


# ---------------------------
# URL & pagination helpers
# ---------------------------

def add_api_params_to_url(url: str, api_key: str, extra_params=None) -> str:
    """
    Safely append api_key, format=json and any extra_params to a URL that
    may already contain query parameters.
    """
    if extra_params is None:
        extra_params = {}

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    # Flatten existing query values & add required params
    merged = {k: v[-1] if isinstance(v, list) else v for k, v in query.items()}
    merged.setdefault("format", "json")
    merged.setdefault("api_key", api_key)

    # Add/override with extra_params
    merged.update(extra_params)

    new_query = urlencode(merged, doseq=False)
    new_parsed = parsed._replace(query=new_query)
    return urlunparse(new_parsed)


def fetch_all_pages_from_url(
    raw_url: str,
    api_key: str,
    params=None,
    delay_sec: float = 0.5,
    max_retries: int = 3,
):
    """
    Generic pagination helper for GameSpot-style endpoints that return:
      - limit
      - offset
      - number_of_page_results
      - number_of_total_results
      - results

    Robust to:
      - Timeouts / connection errors
      - 5xx (including 503 Service Unavailable)
      - JSON decode errors (invalid / truncated JSON)

    On repeated failure for a given endpoint, returns partial results
    instead of crashing the whole run.
    """
    if params is None:
        params = {}

    all_results = []
    offset = 0
    limit = params.get("limit", 100)  # sensible default

    while True:
        page_params = dict(params)
        page_params["offset"] = offset
        page_params["limit"] = limit

        url = add_api_params_to_url(raw_url, api_key, page_params)

        # --- retry loop for this page ---
        attempt = 0
        backoff = 1.0
        last_error = None
        data = None

        while attempt < max_retries:
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=30,
                )

                # Retry explicitly on 503
                if resp.status_code == 503:
                    last_error = f"503 Service Unavailable for {url}"
                    attempt += 1
                    print(f"[WARN] {last_error} (attempt {attempt}/{max_retries}), retrying...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                resp.raise_for_status()

                # JSON decode can fail even if status is 200
                try:
                    data = resp.json()
                except ValueError as e:  # includes JSONDecodeError
                    last_error = f"JSON decode error for {url}: {e}"
                    attempt += 1
                    print(f"[WARN] {last_error} (attempt {attempt}/{max_retries}), retrying...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                # If we got here, we have valid JSON
                break

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_error = f"{type(e).__name__} while fetching {url}: {e}"
                attempt += 1
                print(f"[WARN] {last_error} (attempt {attempt}/{max_retries}), retrying...")
                time.sleep(backoff)
                backoff *= 2

            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status in (500, 502, 503, 504):
                    last_error = f"HTTP {status} while fetching {url}: {e}"
                    attempt += 1
                    print(f"[WARN] {last_error} (attempt {attempt}/{max_retries}), retrying...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    # Non-retryable HTTP error (e.g. 404, 403, 400...)
                    print(f"[ERROR] Non-retryable HTTP error {status} for {url}: {e}")
                    print("[INFO] Returning partial results for this endpoint.")
                    return all_results

        else:
            # Retries exhausted for this page
            print(f"[ERROR] Failed to fetch/parse page after {max_retries} attempts: {raw_url}")
            if last_error:
                print(f"        Last error: {last_error}")
            print("[INFO] Returning partial results for this endpoint.")
            return all_results

        # --- normal pagination logic once we have valid `data` ---
        page_results = data.get("results", []) or []
        all_results.extend(page_results)

        number_of_page_results = data.get("number_of_page_results", len(page_results))
        number_of_total_results = data.get("number_of_total_results", len(page_results))

        if number_of_page_results == 0:
            # No more results
            break

        offset += number_of_page_results
        if offset >= number_of_total_results:
            # Reached the end
            break

        time.sleep(delay_sec)

    return all_results

# ---------------------------
# GameSpot fetchers
# ---------------------------

def fetch_games_by_name(api_key: str, name: str):
    """
    Fetch all games matching the given name from /games/.
    """
    url = f"{BASE_URL}/games/"
    params = {
        "format": "json",
        "api_key": api_key,
        "filter": f"name:{name}",
        "limit": 100,
        "offset": 0,
    }

    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    games = data.get("results", []) or []

    total = data.get("number_of_total_results", len(games))
    page_count = data.get("number_of_page_results", len(games))
    offset = data.get("offset", 0)
    limit = data.get("limit", len(games))

    all_games = list(games)
    while offset + page_count < total:
        offset += page_count
        params["offset"] = offset
        params["limit"] = limit

        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        games = data.get("results", []) or []
        page_count = data.get("number_of_page_results", len(games))
        all_games.extend(games)

        if page_count == 0:
            break

        time.sleep(0.5)

    return all_games


def fetch_related_textual_data_for_game(game_obj: dict, api_key: str):
    """
    Given a single game object from /games/, fetch its related textual data:
      - reviews
      - articles
      - releases
    We intentionally skip purely visual collections (images, videos).
    """
    related = {}

    reviews_url = game_obj.get("reviews_api_url")
    articles_url = game_obj.get("articles_api_url")
    releases_url = game_obj.get("releases_api_url")
    # images_url = game_obj.get("images_api_url")   # skipped (visual)
    # videos_url = game_obj.get("videos_api_url")   # skipped (visual)

    if reviews_url:
        related["reviews"] = fetch_all_pages_from_url(reviews_url, api_key)
    else:
        related["reviews"] = []

    if articles_url:
        related["articles"] = fetch_all_pages_from_url(articles_url, api_key)
    else:
        related["articles"] = []

    if releases_url:
        related["releases"] = fetch_all_pages_from_url(releases_url, api_key)
    else:
        related["releases"] = []

    return related


# ---------------------------
# Main entry point
# ---------------------------

def main():
    # Re-load env to be safe if run as module
    load_dotenv()

    gamespot_api_key = os.getenv("GAMESPOT_API_KEY")
    if not gamespot_api_key:
        raise RuntimeError("Please set GAMESPOT_API_KEY in your .env file")

    # Determine the raw query:
    # - If called like: python -m usage "Dark souls"
    #   use that as input
    # - Otherwise default to "Elden Ring"
    if len(sys.argv) > 1:
        raw_query = " ".join(sys.argv[1:])
    else:
        raw_query = "Elden Ring"

    # Use RAWG to get normalized name
    query_name = resolve_game_name(raw_query)

    print(f"Fetching GameSpot data for '{query_name}' (textual only)...")

    # 1. Fetch all matching games
    games = fetch_games_by_name(gamespot_api_key, query_name)
    print(f"Found {len(games)} game entries for query '{query_name}'.")

    merged = {
        "query": {
            "requested": raw_query,
            "resolved": query_name,
        },
        "games_count": len(games),
        "games": [],
    }

    # 2. For each game, fetch related textual endpoints
    for game in games:
        game_id = game.get("id")
        game_name = game.get("name")
        print(f"Processing game id={game_id}, name={game_name!r}...")

        related_textual = fetch_related_textual_data_for_game(game, gamespot_api_key)

        merged["games"].append(
            {
                "game": game,
                "related": related_textual,
            }
        )

    # 3. Strip all visual-related fields from the final structure
    cleaned = strip_visual_fields(merged)

    # Output file name based on resolved title (optional, safe filename)
    safe_name = query_name.lower().replace(" ", "_")
    output_file = f"{safe_name}_gamespot_full_textual.json"

    # 4. Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved merged textual data to: {output_file}")


if __name__ == "__main__":
    main()
