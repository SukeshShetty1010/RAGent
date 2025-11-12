# usage.py — FIXED & ENHANCED
import json
from data.rawg import RawgTool

def main():
    tool = RawgTool()
    query = "Assassin's Creed Valhalla"

    print(f"\n{'='*80}")
    print(f"RAWG FULL RAG PROFILE: {query}")
    print(f"{'='*80}\n")

    # 1. Ultimate fetch
    profile = tool.fetch_full_profile(query)

    if "error" in profile:
        print("ERROR:", profile)
        return

    # 2. RAG Summary (SAFE)
    print("1. SUMMARY:")
    print(tool.summary(profile["details"]))
    print()

    # 3. RAG Context
    print("2. RAG CONTEXT:")
    print(tool.rag_context(profile["details"]))
    print()

    # 4. Media
    print("3. SCREENSHOTS (first 3):")
    for s in profile["screenshots"][:3]:
        img = s.get("image") or "N/A"
        print(f"   • {img}")
    print()

    print("4. TRAILERS:")
    for m in profile["movies"][:2]:
        name = m.get("name", "Unknown")
        url = m.get("data", {}).get("max", "N/A")
        print(f"   • {name}: {url}")
    print()

    # 5. DLCs
    print("5. DLCs (first 2):")
    for a in profile["additions"][:2]:
        print(f"   • {a.get('name', 'Unknown')} ({a.get('released', 'TBA')})")
    print()

    # 6. Fallback Similar
    print("6. SIMILAR (Free-Tier Fallback):")
    similar = profile["similar_fallback"]
    print(f"   • Suggestions Count: {similar.get('suggestions_count', 0)}")
    print(f"   • Based On: Genres={similar.get('based_on', {}).get('genres', [])}; Tags={similar.get('based_on', {}).get('tags', [])}")
    print()

    # 7. Keys
    print("7. PROFILE KEYS:")
    print(json.dumps(list(profile.keys()), indent=2))

if __name__ == "__main__":
    main()