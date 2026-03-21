#!/usr/bin/env python3
"""
scripts/bulk_ingest.py — Production-Grade Bulk Ingestion Pipeline

Processes up to 300 games through the RAGent ingestion pipeline (Stages 1-5).
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
# TOP 300 GAMES DATASET
# Curated from: IGN Top 100, Metacritic All-Time, Wikipedia Best-Selling,
# Google Year-in-Search 2024/2025, GameSpot Archives, RAWG database.
# Grouped by genre for clean data formatting.
# =====================================================================

TOP_300_GAMES: List[str] = [
    # ── Action-Adventure (40) ──────────────────────────────────────
    "The Legend of Zelda: Breath of the Wild",
    "The Legend of Zelda: Tears of the Kingdom",
    "The Legend of Zelda: Ocarina of Time",
    "Red Dead Redemption 2",
    "Red Dead Redemption",
    "Grand Theft Auto V",
    "Grand Theft Auto IV",
    "Grand Theft Auto: San Andreas",
    "Grand Theft Auto: Vice City",
    "The Last of Us",
    "The Last of Us Part II",
    "God of War",
    "God of War Ragnarok",
    "Uncharted 4: A Thief's End",
    "Uncharted 2: Among Thieves",
    "Horizon Zero Dawn",
    "Horizon Forbidden West",
    "Ghost of Tsushima",
    "Spider-Man",
    "Spider-Man: Miles Morales",
    "Batman: Arkham City",
    "Batman: Arkham Knight",
    "Assassin's Creed Valhalla",
    "Assassin's Creed Odyssey",
    "Assassin's Creed Origins",
    "Assassin's Creed II",
    "Assassin's Creed Brotherhood",
    "Assassin's Creed IV: Black Flag",
    "Far Cry 5",
    "Far Cry 3",
    "Far Cry 6",
    "Tomb Raider (2013)",
    "Rise of the Tomb Raider",
    "Shadow of the Tomb Raider",
    "Death Stranding",
    "Death Stranding 2",
    "Metal Gear Solid V: The Phantom Pain",
    "Metal Gear Solid 3: Snake Eater",
    "Sekiro: Shadows Die Twice",
    "Star Wars Jedi: Fallen Order",

    # ── RPG / JRPG (45) ───────────────────────────────────────────
    "The Witcher 3: Wild Hunt",
    "The Elder Scrolls V: Skyrim",
    "The Elder Scrolls IV: Oblivion",
    "Elden Ring",
    "Dark Souls III",
    "Dark Souls",
    "Dark Souls II",
    "Bloodborne",
    "Baldur's Gate 3",
    "Cyberpunk 2077",
    "Mass Effect 2",
    "Mass Effect 3",
    "Mass Effect Legendary Edition",
    "Fallout 4",
    "Fallout: New Vegas",
    "Fallout 3",
    "Dragon Age: Inquisition",
    "Dragon Age: Origins",
    "Final Fantasy VII",
    "Final Fantasy VII Remake",
    "Final Fantasy VII Rebirth",
    "Final Fantasy X",
    "Final Fantasy XV",
    "Final Fantasy XVI",
    "Persona 5",
    "Persona 5 Royal",
    "Persona 4 Golden",
    "Persona 3 Reload",
    "Xenoblade Chronicles 3",
    "Xenoblade Chronicles 2",
    "Chrono Trigger",
    "Kingdom Hearts",
    "Kingdom Hearts II",
    "NieR: Automata",
    "NieR Replicant",
    "Divinity: Original Sin 2",
    "Disco Elysium",
    "Dragon's Dogma 2",
    "Diablo IV",
    "Diablo III",
    "Diablo II: Resurrected",
    "Path of Exile 2",
    "Monster Hunter: World",
    "Monster Hunter Rise",
    "Monster Hunter Wilds",

    # ── Shooter / FPS (35) ─────────────────────────────────────────
    "Call of Duty: Modern Warfare",
    "Call of Duty: Modern Warfare 2 (2022)",
    "Call of Duty: Modern Warfare II",
    "Call of Duty: Black Ops",
    "Call of Duty: Black Ops II",
    "Call of Duty: Black Ops III",
    "Call of Duty: Black Ops 6",
    "Call of Duty: Warzone",
    "Halo Infinite",
    "Halo 3",
    "Halo: Reach",
    "Halo: Combat Evolved",
    "Doom Eternal",
    "Doom (2016)",
    "DOOM: The Dark Ages",
    "Half-Life 2",
    "Half-Life: Alyx",
    "Counter-Strike 2",
    "Counter-Strike: Global Offensive",
    "Destiny 2",
    "Destiny",
    "Titanfall 2",
    "Borderlands 3",
    "Borderlands 2",
    "BioShock Infinite",
    "BioShock",
    "Far Cry 4",
    "Resident Evil 4 (2023)",
    "Resident Evil Village",
    "Resident Evil 2 (2019)",
    "Left 4 Dead 2",
    "Overwatch 2",
    "Overwatch",
    "Rainbow Six Siege",
    "Helldivers 2",

    # ── Battle Royale / Multiplayer (15) ───────────────────────────
    "Fortnite",
    "PUBG: Battlegrounds",
    "Apex Legends",
    "Warzone 2.0",
    "Fall Guys",
    "Valorant",
    "Palworld",
    "Among Us",
    "Lethal Company",
    "Phasmophobia",
    "Dead by Daylight",
    "Rust",
    "Escape from Tarkov",
    "The Finals",
    "ARC Raiders",

    # ── Open World / Sandbox (20) ──────────────────────────────────
    "Minecraft",
    "Terraria",
    "Stardew Valley",
    "Animal Crossing: New Horizons",
    "No Man's Sky",
    "Starfield",
    "Subnautica",
    "Subnautica: Below Zero",
    "Satisfactory",
    "Valheim",
    "Ark: Survival Evolved",
    "Ark: Survival Ascended",
    "Astroneer",
    "The Forest",
    "Sons of the Forest",
    "Grounded",
    "Deep Rock Galactic",
    "Conan Exiles",
    "Raft",
    "7 Days to Die",

    # ── Platformer / Indie (25) ────────────────────────────────────
    "Hollow Knight",
    "Hollow Knight: Silksong",
    "Celeste",
    "Hades",
    "Hades II",
    "Cuphead",
    "Ori and the Will of the Wisps",
    "Ori and the Blind Forest",
    "Super Mario Odyssey",
    "Super Mario Galaxy",
    "Super Mario Bros. Wonder",
    "Donkey Kong Country: Tropical Freeze",
    "Rayman Legends",
    "Shovel Knight",
    "Dead Cells",
    "It Takes Two",
    "A Way Out",
    "Sackboy: A Big Adventure",
    "Little Nightmares II",
    "Inside",
    "Limbo",
    "Undertale",
    "Outer Wilds",
    "Return of the Obra Dinn",
    "Braid",

    # ── Strategy / Simulation / City-Builder (25) ──────────────────
    "Civilization VI",
    "Civilization V",
    "Age of Empires IV",
    "Age of Empires II: Definitive Edition",
    "Total War: Warhammer III",
    "Total War: Three Kingdoms",
    "XCOM 2",
    "Fire Emblem: Three Houses",
    "Fire Emblem Engage",
    "Crusader Kings III",
    "Stellaris",
    "Europa Universalis IV",
    "Cities: Skylines",
    "Cities: Skylines II",
    "Planet Coaster",
    "Planet Zoo",
    "Factorio",
    "RimWorld",
    "Frostpunk",
    "Frostpunk 2",
    "Two Point Hospital",
    "Sims 4",
    "SimCity (2013)",
    "Tropico 6",
    "Dwarf Fortress",

    # ── Sports / Racing (20) ───────────────────────────────────────
    "FIFA 23",
    "EA Sports FC 24",
    "EA Sports FC 25",
    "EA Sports College Football 25",
    "NBA 2K24",
    "NBA 2K25",
    "Madden NFL 24",
    "Madden NFL 25",
    "MLB The Show 24",
    "Forza Horizon 5",
    "Forza Motorsport (2023)",
    "Gran Turismo 7",
    "Need for Speed: Unbound",
    "Need for Speed: Heat",
    "Mario Kart 8 Deluxe",
    "Rocket League",
    "F1 24",
    "Riders Republic",
    "Tony Hawk's Pro Skater 1 + 2",
    "Wii Sports",

    # ── Fighting (10) ──────────────────────────────────────────────
    "Street Fighter 6",
    "Mortal Kombat 1 (2023)",
    "Mortal Kombat 11",
    "Tekken 8",
    "Super Smash Bros. Ultimate",
    "Dragon Ball FighterZ",
    "Dragon Ball: Sparking! Zero",
    "Guilty Gear Strive",
    "Injustice 2",
    "MultiVersus",

    # ── Horror / Survival (15) ─────────────────────────────────────
    "Resident Evil 4",
    "Silent Hill 2 (2024)",
    "Alan Wake 2",
    "Amnesia: The Dark Descent",
    "Outlast",
    "Outlast 2",
    "The Evil Within 2",
    "Alien: Isolation",
    "Dead Space (2023)",
    "Days Gone",
    "Dying Light 2",
    "Dying Light",
    "The Callisto Protocol",
    "Until Dawn",
    "Blair Witch",

    # ── MMORPG / Live Service (15) ─────────────────────────────────
    "World of Warcraft",
    "Final Fantasy XIV",
    "Guild Wars 2",
    "Elder Scrolls Online",
    "Lost Ark",
    "New World",
    "Genshin Impact",
    "Honkai: Star Rail",
    "Wuthering Waves",
    "Tower of Fantasy",
    "Warframe",
    "Diablo Immortal",
    "Black Desert Online",
    "Phantasy Star Online 2",
    "Destiny 2: The Final Shape",

    # ── Narrative / Walking Sim / Puzzle (20) ──────────────────────
    "Portal 2",
    "Portal",
    "Life is Strange",
    "Life is Strange: True Colors",
    "Detroit: Become Human",
    "Heavy Rain",
    "Firewatch",
    "What Remains of Edith Finch",
    "The Stanley Parable: Ultra Deluxe",
    "Clair Obscur: Expedition 33",
    "Psychonauts 2",
    "A Plague Tale: Requiem",
    "A Plague Tale: Innocence",
    "Control",
    "Deathloop",
    "Twelve Minutes",
    "The Witness",
    "Obra Dinn",
    "Inscryption",
    "Split Fiction",

    # ── Mobile / Casual Crossover (10) ─────────────────────────────
    "Pokémon GO",
    "Pokémon Legends: Arceus",
    "Pokémon Legends: Z-A",
    "Pokémon Scarlet and Violet",
    "Clash Royale",
    "Roblox",
    "Candy Crush Saga",
    "Genshin Impact",
    "Black Myth: Wukong",
    "Infinite Craft",

    # ── Retro / Classic Essentials (15) ────────────────────────────
    "Tetris",
    "Pac-Man",
    "Super Mario Bros.",
    "Sonic the Hedgehog",
    "Sonic the Hedgehog 2",
    "Sonic Frontiers",
    "Metroid Dread",
    "Metroid Prime Remastered",
    "Metroid Prime 4: Beyond",
    "Mega Man X",
    "Castlevania: Symphony of the Night",
    "Street Fighter II",
    "Doom (1993)",
    "Super Metroid",
    "The Legend of Zelda: A Link to the Past",
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
        description="Bulk-ingest up to 300 games into RAGent's Qdrant database.",
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
        default=len(TOP_300_GAMES),
        help=f"End index (exclusive, default: {len(TOP_300_GAMES)})",
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
    games_slice = TOP_300_GAMES[args.start : args.end]
    total = len(games_slice)

    if not games_slice:
        logger.error("No games in range [%d:%d]. Total available: %d",
                      args.start, args.end, len(TOP_300_GAMES))
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
