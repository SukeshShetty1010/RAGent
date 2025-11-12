#!/usr/bin/env python3
"""
usage.py — Quick test script for IGDB + RAWG integration.
Checks:
1. RAWG name correction
2. IGDB authentication
3. IGDB text-focused data retrieval
"""

import json
from data.igdb_data import IGDBData

def main():
    print("=" * 80)
    print("🎮  RAGent IGDB Data Fetch Test")
    print("=" * 80)

    query = input("Enter game title or keyword: ").strip()
    if not query:
        print("No input provided. Exiting.")
        return

    print(f"\nFetching structured IGDB data for '{query}'...\n")

    try:
        data = IGDBData()
        result = data.get_game_by_name(query)

        if not result:
            print("⚠️  No data found.")
            return

        print("\n✅ SUCCESS — Cleaned, Textual IGDB Data:\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print("\n❌ ERROR:")
        print(e)

if __name__ == "__main__":
    main()
