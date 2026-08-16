#!/usr/bin/env python3
"""
scripts/check_sources.py

Source health check for the ingestion pipeline. Run before and after
every bulk rebuild batch to confirm what's actually live before burning
API budget on a host that's down.

Prints one status line per source. Never prints key/secret *values* —
only whether a key is set and its length, so this is safe to paste into
a bug report or CI log.

Exit code is non-zero only when the rebuild cannot proceed at all:
  - Qdrant unreachable (hard dependency — nothing works without it)
  - Neither IGDB nor RAWG available (no identity resolution possible)
  - GameSpot, Wikipedia, and Steam all down (empty editorial corpus)
A single dead source among several redundant ones is NOT a failure —
that's the point of the multi-provider design.

Usage:
    python -m scripts.check_sources
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Windows consoles/redirected-output pipes often default to cp1252, which
# can't encode the checkmark/cross below — force UTF-8 so this never
# crashes on the summary line regardless of where stdout is going.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROBE_GAME = "Elden Ring"


def _key_status(env_var: str) -> str:
    value = os.getenv(env_var)
    return f"set (len={len(value)})" if value else "MISSING"


def check_qdrant() -> bool:
    from qdrant_client import QdrantClient

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")

    try:
        client = QdrantClient(url=url, api_key=api_key or None, timeout=10)
        collections = client.get_collections().collections
        names = ", ".join(c.name for c in collections) or "(none)"
        print(f"[OK]   Qdrant       — {len(collections)} collections: {names}")
        client.close()
        return True
    except Exception as exc:
        print(f"[DOWN] Qdrant       — {type(exc).__name__}: {str(exc)[:150]}")
        return False


def check_rawg() -> bool:
    from data.rawg_data import RawgError, fetch_rawg_game_data, rawg_available

    if not rawg_available():
        print("[OFF]  RAWG         — disabled or circuit open (skipped)")
        return False

    try:
        fetch_rawg_game_data(PROBE_GAME)
        print("[OK]   RAWG         — reachable")
        return True
    except RawgError as exc:
        print(f"[DOWN] RAWG         — {str(exc)[:150]}")
        return False
    except Exception as exc:
        print(f"[DOWN] RAWG         — {type(exc).__name__}: {str(exc)[:150]}")
        return False


def check_igdb() -> bool:
    from data.igdb_data import fetch_igdb_game_data

    try:
        result = fetch_igdb_game_data(PROBE_GAME, strip_visual=True, limit=1)
        count = len(result.get("clean") or [])
        print(f"[OK]   IGDB         — {count} result(s) for probe query")
        return count > 0
    except Exception as exc:
        print(f"[DOWN] IGDB         — {type(exc).__name__}: {str(exc)[:150]}")
        return False


def check_gamespot() -> bool:
    from data.gamespot_data import fetch_games_by_name

    api_key = os.getenv("GAMESPOT_API_KEY")
    if not api_key:
        print("[OFF]  GameSpot     — GAMESPOT_API_KEY not set")
        return False

    try:
        games = fetch_games_by_name(api_key, PROBE_GAME, limit=1)
        if games:
            print(f"[OK]   GameSpot     — {len(games)} result(s) for probe query")
            return True
        print("[DOWN] GameSpot     — 0 results (likely Cloudflare bot-protection block)")
        return False
    except Exception as exc:
        print(f"[DOWN] GameSpot     — {type(exc).__name__}: {str(exc)[:150]}")
        return False


def check_wikipedia() -> bool:
    from data.wikipedia_data import fetch_wikipedia_article

    try:
        article = fetch_wikipedia_article(PROBE_GAME)
        if article:
            print(f"[OK]   Wikipedia    — {len(article['extract'])} chars for probe query")
            return True
        print("[DOWN] Wikipedia    — no article resolved for probe query")
        return False
    except Exception as exc:
        print(f"[DOWN] Wikipedia    — {type(exc).__name__}: {str(exc)[:150]}")
        return False


def check_steam() -> bool:
    from data.steam_data import fetch_steam_game_data

    try:
        data = fetch_steam_game_data(PROBE_GAME)
        if data:
            print(f"[OK]   Steam        — resolved appid {data.get('_appid')}")
            return True
        print("[DOWN] Steam        — no app resolved for probe query")
        return False
    except Exception as exc:
        print(f"[DOWN] Steam        — {type(exc).__name__}: {str(exc)[:150]}")
        return False


def check_tavily() -> bool:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("[OFF]  Tavily       — TAVILY_API_KEY not set")
        return False

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        result = client.search(query=f"{PROBE_GAME} review", max_results=1)
        count = len(result.get("results") or [])
        print(f"[OK]   Tavily       — {count} result(s) for probe query")
        return count > 0
    except Exception as exc:
        print(f"[DOWN] Tavily       — {type(exc).__name__}: {str(exc)[:150]}")
        return False


def main() -> None:
    print("=== Environment key presence ===")
    for env_var in (
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "RAWG_API_KEY",
        "TWITCH_CLIENT_ID",
        "TWITCH_CLIENT_SECRET",
        "GAMESPOT_API_KEY",
        "TAVILY_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
    ):
        print(f"  {env_var}: {_key_status(env_var)}")

    print(f"\n=== Live source probe (query: {PROBE_GAME!r}) ===")
    qdrant_ok = check_qdrant()
    rawg_ok = check_rawg()
    igdb_ok = check_igdb()
    gamespot_ok = check_gamespot()
    wikipedia_ok = check_wikipedia()
    steam_ok = check_steam()
    tavily_ok = check_tavily()

    identity_ok = igdb_ok or rawg_ok
    editorial_ok = gamespot_ok or wikipedia_ok or steam_ok

    print("\n=== Summary ===")
    print(f"  Qdrant reachable:            {'YES' if qdrant_ok else 'NO'}")
    print(f"  Identity resolution possible: {'YES' if identity_ok else 'NO'} (IGDB or RAWG)")
    print(
        f"  Editorial corpus possible:    {'YES' if editorial_ok else 'NO'} "
        "(GameSpot, Wikipedia, or Steam)"
    )
    print(f"  Tavily web-search fallback:   {'YES' if tavily_ok else 'NO'}")

    if not qdrant_ok:
        print("\n❌ BLOCKING: Qdrant is unreachable — nothing can be ingested or queried.")
        sys.exit(1)

    if not identity_ok:
        print("\n❌ BLOCKING: neither IGDB nor RAWG is available — no identity resolution.")
        sys.exit(1)

    if not editorial_ok:
        print(
            "\n❌ BLOCKING: GameSpot, Wikipedia, and Steam are all down — "
            "rebuild would produce an empty EditorialChunk corpus."
        )
        sys.exit(1)

    print("\n✅ Rebuild can proceed.")


if __name__ == "__main__":
    main()
