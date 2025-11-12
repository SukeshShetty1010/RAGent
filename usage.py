# usage.py
"""
================================================================================
🎮  RAGent GameSpot Data Fetch Test
================================================================================
Quick interactive script to test the GameSpot client and data module.
================================================================================
"""

from data.gamespot_data import GameSpotData
import json

def main():
    print("=" * 80)
    print("🎮  RAGent GameSpot Data Fetch Test")
    print("=" * 80)

    title = input("Enter game title or keyword: ").strip()
    if not title:
        print("❌ Please provide a valid title.")
        return

    print(f"\nFetching structured GameSpot data for '{title}'...\n")

    data = GameSpotData()
    structured = data.get_game_data(title)

    if not structured:
        print("⚠️  No data found for that game.")
        return

    # Display a quick preview
    print("\n✅ SUCCESS — Structured hierarchical data retrieved!\n")
    print(json.dumps(structured, indent=2, ensure_ascii=False))

    # Save result locally
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    output_path = f"gamespot_structured_{safe_title}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Data saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
