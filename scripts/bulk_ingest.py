#!/usr/bin/env python3
"""
scripts/bulk_ingest.py — Production-Grade Bulk Ingestion Pipeline

Processes up to 100 curated top-grossing / most-searched games through
the RAGent ingestion pipeline (Stages 1-5).
Features:
  - Rate limiting (2.0s inter-game cooldown)
  - Fault tolerance (per-game try/except, never crashes on single failure)
  - Resume support (reads success_games.log to skip already-processed games)
  - Dry-run mode (prints game list without making API calls)
  - Range slicing (--start / --end for batched execution)
  - Structured logging to failed_games.log and success_games.log

Usage:
    python scripts/bulk_ingest.py --dry-run
    python scripts/bulk_ingest.py --start 0 --end 10
    python scripts/bulk_ingest.py                      # all 300 games
    python scripts/bulk_ingest.py --resume              # skip already-done games
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

from dotenv import load_dotenv

# ── Load .env before ANY other imports that use os.getenv ──
load_dotenv()

from qdrant_client import QdrantClient

from upsert.upsert_canonical_game import upsert_game_anchor
from upsert.upsert_platform_specs import upsert_platform_specs
from upsert.upsert_igdb_metadata import upsert_igdb_context
from upsert.upsert_gamespot_chunks import upsert_gamespot_container
from embed.prepare_editorial_payloads import generate_chunk_payloads
from upsert.upsert_editorial_chunks import upsert_chunk_batch

# =====================================================================
# CONFIGURATION
# =====================================================================

INTER_GAME_DELAY: float = 2.0   # seconds between games (rate limiting)
LOG_DIR: Path = Path("logs")
SUCCESS_LOG: Path = LOG_DIR / "success_games.log"
FAILED_LOG: Path = LOG_DIR / "failed_games.log"

# =====================================================================
# LOGGING SETUP
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bulk_ingest")

# =====================================================================
# TOP 100 GAMES DATASET
# Curated from: Wikipedia Best-Selling Games, Metacritic All-Time Best,
# Google Year-in-Search 2024/2025, Steam Charts, GameSpot Archives.
# Selected for commercial success, review coverage, and search volume.
# Grouped by genre for clean data distribution.
# =====================================================================

TOP_100_GAMES: List[str] = [

    # ── Action-Adventure (15) ─────────────────────────────────────────
    "Grand Theft Auto V",
    "Red Dead Redemption 2",
    "The Legend of Zelda: Breath of the Wild",
    "The Legend of Zelda: Tears of the Kingdom",
    "The Last of Us",
    "The Last of Us Part II",
    "God of War",
    "God of War Ragnarok",
    "Ghost of Tsushima",
    "Spider-Man",
    "Spider-Man: Miles Morales",
    "Assassin's Creed Valhalla",
    "Assassin's Creed Odyssey",
    "Far Cry 5",
    "Death Stranding",

    # ── RPG / JRPG (15) ───────────────────────────────────────────────
    "The Witcher 3: Wild Hunt",
    "Elden Ring",
    "Cyberpunk 2077",
    "Baldur's Gate 3",
    "The Elder Scrolls V: Skyrim",
    "Fallout 4",
    "Fallout: New Vegas",
    "Final Fantasy VII Remake",
    "Persona 5 Royal",
    "Dark Souls III",
    "Bloodborne",
    "NieR: Automata",
    "Monster Hunter: World",
    "Diablo IV",
    "Dragon's Dogma 2",

    # ── Shooter / FPS (15) ────────────────────────────────────────────
    "Call of Duty: Modern Warfare",
    "Call of Duty: Black Ops II",
    "Call of Duty: Warzone",
    "Halo 3",
    "Halo Infinite",
    "Doom Eternal",
    "Half-Life 2",
    "Counter-Strike: Global Offensive",
    "Counter-Strike 2",
    "Destiny 2",
    "Titanfall 2",
    "BioShock Infinite",
    "Borderlands 3",
    "Resident Evil 4 (2023)",
    "Helldivers 2",

    # ── Battle Royale / Multiplayer (8) ───────────────────────────────
    "Fortnite",
    "PUBG: Battlegrounds",
    "Apex Legends",
    "Valorant",
    "Palworld",
    "Among Us",
    "Rust",
    "Fall Guys",

    # ── Open World / Sandbox (8) ──────────────────────────────────────
    "Minecraft",
    "Stardew Valley",
    "Animal Crossing: New Horizons",
    "No Man's Sky",
    "Terraria",
    "Valheim",
    "Subnautica",
    "Starfield",

    # ── Platformer / Indie (8) ────────────────────────────────────────
    "Hollow Knight",
    "Hades",
    "Celeste",
    "Cuphead",
    "Super Mario Odyssey",
    "It Takes Two",
    "Ori and the Will of the Wisps",
    "Undertale",

    # ── Strategy / Simulation (7) ─────────────────────────────────────
    "Civilization VI",
    "Age of Empires IV",
    "Crusader Kings III",
    "Factorio",
    "RimWorld",
    "Cities: Skylines",
    "Stellaris",

    # ── Sports / Racing (5) ───────────────────────────────────────────
    "EA Sports FC 24",
    "Forza Horizon 5",
    "Gran Turismo 7",
    "Mario Kart 8 Deluxe",
    "Rocket League",

    # ── Fighting (5) ──────────────────────────────────────────────────
    "Street Fighter 6",
    "Mortal Kombat 11",
    "Tekken 8",
    "Super Smash Bros. Ultimate",
    "Dragon Ball FighterZ",

    # ── Horror / Survival (5) ─────────────────────────────────────────
    "Resident Evil Village",
    "Resident Evil 2 (2019)",
    "Alan Wake 2",
    "Alien: Isolation",
    "Dead Space (2023)",

    # ── MMORPG / Live Service (5) ─────────────────────────────────────
    "World of Warcraft",
    "Final Fantasy XIV",
    "Genshin Impact",
    "Honkai: Star Rail",
    "Warframe",

    # ── Narrative / Puzzle (4) ────────────────────────────────────────
    "Portal 2",
    "Detroit: Become Human",
    "A Plague Tale: Requiem",
    "Control",
]

# =====================================================================
# HELPERS
# =====================================================================


def _get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client from environment variables."""
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY", "")
    return QdrantClient(url=url, api_key=api_key or None)


def _load_success_set() -> Set[str]:
    """Load set of already-processed game names from success log."""
    if not SUCCESS_LOG.exists():
        return set()
    lines = SUCCESS_LOG.read_text(encoding="utf-8").strip().splitlines()
    # Each line: "TIMESTAMP | game_name"
    games: Set[str] = set()
    for line in lines:
        parts = line.split("|", maxsplit=1)
        if len(parts) == 2:
            games.add(parts[1].strip())
    return games


def _log_result(filepath: Path, game_name: str, detail: str = "") -> None:
    """Append a timestamped entry to a log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"{ts} | {game_name}"
    if detail:
        entry += f" | {detail}"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def _ingest_single_game(client: QdrantClient, game_name: str) -> None:
    """
    Run the full 5-stage ingestion pipeline for a single game.

    Mirrors the logic in upsert/upsert_all.py:main() but without
    argparse or sys.exit, so failures are catchable.
    """

    # Stage 1: Canonical Game Anchor
    logger.info("  Stage 1: Canonical Game Anchor...")
    game_uuid = upsert_game_anchor(client, game_name)
    if not game_uuid or not isinstance(game_uuid, str):
        raise RuntimeError("Invalid UUID from upsert_game_anchor")
    logger.info("    ✅ Anchor UUID: %s", game_uuid)

    # Stage 2: Platform Specs
    logger.info("  Stage 2: Platform Specs...")
    spec_count = upsert_platform_specs(
        client=client, game_name=game_name, game_uuid=game_uuid
    )
    logger.info("    ✅ Upserted %d platform specs", spec_count)

    # Stage 3: IGDB Metadata
    logger.info("  Stage 3: IGDB Metadata...")
    igdb_count = upsert_igdb_context(
        client=client, game_title=game_name, game_uuid=game_uuid
    )
    logger.info("    ✅ Upserted %d IGDB entities", igdb_count)

    # Stage 4: GameSpot Editorial Container (fail-soft)
    logger.info("  Stage 4: GameSpot Editorial Container...")
    try:
        gamespot_uuid = upsert_gamespot_container(
            client=client, game_name=game_name, game_uuid=game_uuid
        )
        if gamespot_uuid:
            logger.info("    ✅ GameSpot UUID: %s", gamespot_uuid)
        else:
            logger.warning("    ⚠️  No GameSpot data found — skipped")
    except Exception as exc:
        logger.warning("    ⚠️  Stage 4 failed (non-fatal): %s", exc)

    # Stage 5: Editorial Chunking + Embedding (fail-soft)
    logger.info("  Stage 5: Editorial Chunking & Embedding...")
    try:
        chunks = generate_chunk_payloads(game_name, game_uuid)
        if chunks:
            upsert_chunk_batch(chunks)
            logger.info("    ✅ Upserted %d editorial chunks", len(chunks))
        else:
            logger.warning("    ⚠️  No editorial chunks generated — skipped")
    except Exception as exc:
        logger.warning("    ⚠️  Stage 5 failed (non-fatal): %s", exc)


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-ingest up to 100 curated games into RAGent's Qdrant database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the games list (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=len(TOP_100_GAMES),
        help=f"End index (exclusive, default: {len(TOP_100_GAMES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the games list without making any API calls",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip games already logged in success_games.log",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=INTER_GAME_DELAY,
        help=f"Seconds between games (default: {INTER_GAME_DELAY})",
    )
    args = parser.parse_args()

    # ── Validate range ──
    games_slice = TOP_100_GAMES[args.start : args.end]
    total = len(games_slice)

    if not games_slice:
        logger.error("No games in range [%d:%d]. Total available: %d",
                      args.start, args.end, len(TOP_100_GAMES))
        sys.exit(1)

    # ── Dry run ──
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN — {total} games in range [{args.start}:{args.end}]")
        print(f"{'='*60}\n")
        for i, name in enumerate(games_slice, start=args.start):
            print(f"  [{i:3d}] {name}")
        print(f"\n  Total: {total} games")
        print(f"  Estimated time: ~{total * (args.delay + 30):.0f}s "
              f"({total * (args.delay + 30) / 60:.1f} min)")
        return

    # ── Resume support ──
    already_done: Set[str] = set()
    if args.resume:
        already_done = _load_success_set()
        logger.info("Resume mode: %d games already processed", len(already_done))

    # ── Connect to Qdrant ──
    logger.info("Connecting to Qdrant...")
    client = _get_qdrant_client()

    succeeded = 0
    failed = 0
    skipped = 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for idx, game_name in enumerate(games_slice, start=args.start):
            # ── Skip if already done ──
            if game_name in already_done:
                logger.info("[%3d/%3d] SKIP (already done): %s",
                            idx + 1, args.start + total, game_name)
                skipped += 1
                continue

            logger.info("=" * 60)
            logger.info("[%3d/%3d] INGESTING: %s",
                        idx + 1, args.start + total, game_name)
            logger.info("=" * 60)

            try:
                _ingest_single_game(client, game_name)
                succeeded += 1
                _log_result(SUCCESS_LOG, game_name)
                logger.info("✅ SUCCESS: %s", game_name)

            except Exception as exc:
                failed += 1
                error_msg = f"{type(exc).__name__}: {exc}"
                _log_result(FAILED_LOG, game_name, error_msg)
                logger.error("❌ FAILED: %s — %s", game_name, error_msg)

            # ── Rate limiting ──
            if idx < args.start + total - 1:
                logger.info("⏳ Cooling down %.1fs...", args.delay)
                time.sleep(args.delay)

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user. Progress saved to logs.")

    finally:
        client.close()
        logger.info("")
        logger.info("=" * 60)
        logger.info("  BULK INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info("  Total in range:  %d", total)
        logger.info("  Succeeded:       %d", succeeded)
        logger.info("  Failed:          %d", failed)
        logger.info("  Skipped (resume):%d", skipped)
        logger.info("  Success log:     %s", SUCCESS_LOG)
        logger.info("  Failed log:      %s", FAILED_LOG)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
