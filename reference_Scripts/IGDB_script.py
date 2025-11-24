import os
import json
import time
import requests
from dotenv import load_dotenv

# ======================================
# Load environment variables
# ======================================

load_dotenv()

# RAWG API
RAWG_API_KEY = os.getenv("RAWG_API_KEY")

# IGDB API
TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IGDB_GAMES_URL = "https://api.igdb.com/v4/games"

# Visual/media keys to remove
VISUAL_KEYS = {"artworks", "cover", "screenshots", "videos"}


# ======================================
# RAWG — Find Correct Game Name
# ======================================

def get_correct_game_name(query: str) -> str | None:
    """Return the official game name using RAWG search."""

    if not RAWG_API_KEY:
        raise ValueError("Missing RAWG_API_KEY in .env file!")

    url = "https://api.rawg.io/api/games"
    params = {
        "key": RAWG_API_KEY,
        "search": query,
        "page_size": 1,
    }

    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()

        if data.get("results"):
            return data["results"][0].get("name")

        return None

    except Exception as e:
        print(f"RAWG error: {e}")
        return None


# ======================================
# IGDB — Authentication + Request
# ======================================

def get_twitch_token() -> str:
    """Get OAuth token for IGDB."""
    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": TWITCH_CLIENT_ID,
            "client_secret": TWITCH_CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def igdb_fetch(query: str, token: str):
    """POST to IGDB /games with the given query."""
    headers = {
        "Client-ID": TWITCH_CLIENT_ID,
        "Authorization": f"Bearer {token}",
    }
    resp = requests.post(IGDB_GAMES_URL, headers=headers, data=query, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ======================================
# Cleaning logic (strip visual fields)
# ======================================

def strip_visual_fields(records, visual_keys):
    cleaned = []
    for item in records:
        cleaned_item = {k: v for k, v in item.items() if k not in visual_keys}
        cleaned.append(cleaned_item)
    return cleaned


# ======================================
# Main workflow
# ======================================

def full_pipeline(user_input_name: str):
    # 1. RAWG → Correct name
    print(f"🔍 Searching RAWG for correct name for: {user_input_name}")
    correct_name = get_correct_game_name(user_input_name)

    if not correct_name:
        print("❌ No matching game found on RAWG.")
        return

    print(f"✔ Correct RAWG name found: {correct_name}")

    # 2. IGDB → OAuth Token
    print("🔐 Getting Twitch/IGDB auth token...")
    token = get_twitch_token()
    print("✔ Token received.")

    # 3. IGDB Query (fields *)
    print("🎮 Fetching FULL IGDB data...")
    query = f"""
        fields *;
        search "{correct_name}";
        limit 500;
    """

    data = igdb_fetch(query=query, token=token)
    print(f"✔ Retrieved {len(data)} game records from IGDB.")

    # File names
    timestamp = int(time.time())
    raw_file = f"{correct_name.lower().replace(' ', '_')}_igdb_full_{timestamp}.json"
    clean_file = f"{correct_name.lower().replace(' ', '_')}_igdb_clean_{timestamp}.json"

    # 4. Save raw data
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Raw IGDB data saved → {raw_file}")

    # 5. Strip visual fields
    cleaned_data = strip_visual_fields(data, VISUAL_KEYS)

    with open(clean_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    print(f"💾 Cleaned IGDB metadata saved → {clean_file}")

    print("\n🎉 Pipeline complete!")
    print("Raw file :", raw_file)
    print("Clean file:", clean_file)


# ======================================
# Run
# ======================================

if __name__ == "__main__":
    game_input = input("Enter game name: ")
    full_pipeline(game_input)
