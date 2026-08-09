"""
PlatformSpec Ingestion Module

Extracts platform-specific operational data from cleaned RAWG output
and prepares deterministic, upsert-ready PlatformSpec payloads
for Qdrant.

Responsibilities:
- Isolate operational (non-semantic) platform data
- Generate deterministic UUIDs to avoid duplication
- Hard-link each PlatformSpec to its canonical Game
"""

from typing import Any, Dict, List
from datetime import datetime

from uuid import UUID, uuid5


# ------------------------------------------------------------------
# Helpers (intentionally local and minimal)
# ------------------------------------------------------------------

def _safe_iso_date(value: Any) -> str | None:
    """
    Convert a date/datetime-like value to ISO YYYY-MM-DD.
    Returns None if parsing fails.
    """
    if not value or not isinstance(value, str):
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value[: len(fmt)], fmt).date().isoformat()
        except Exception:
            continue

    return None


# ------------------------------------------------------------------
# Core payload generator
# ------------------------------------------------------------------

def generate_platform_payloads(
    cleaned_data: Dict[str, Any],
    game_uuid: str,
) -> List[Dict[str, Any]]:
    """
    Generate deterministic PlatformSpec payloads.

    Args:
        cleaned_data: RAWG-cleaned game dictionary
        game_uuid: UUID of the canonical Game object (Weaviate)

    Returns:
        List of dictionaries ready for Weaviate batch insertion
    """

    platforms = cleaned_data.get("platforms") or []

    payloads: List[Dict[str, Any]] = []

    for p in platforms:
        platform_name = p.get("platform_name")
        if not platform_name:
            # Platform without a name is meaningless; skip defensively
            continue

        # ------------------------------------------------------------------
        # Deterministic UUID
        # Seed: "<game_uuid>_<platform_name>" — provider-independent
        # (previously seeded from RAWG's integer game_id, which vendor-
        # locked PlatformSpec identity the same way the Game anchor used
        # to be before ingest/identity_resolver.py's provider-independent
        # resolution; game_uuid is stable across providers).
        # ------------------------------------------------------------------
        uuid_seed = f"{game_uuid}_{platform_name}"
        _NS = UUID("12345678-1234-5678-1234-567812345678")
        platform_uuid = str(uuid5(_NS, uuid_seed))

        payload = {
            "uuid": platform_uuid,
            "class": "PlatformSpec",
            "properties": {
                "platform_name": platform_name,
                "platform_family": p.get("platform_family"),
                "release_date": _safe_iso_date(p.get("release_date")),
                "requirements_minimum": p.get("requirements_minimum"),
                "requirements_recommended": p.get("requirements_recommended"),
                # ----------------------------------------------------------
                # Hard reference to canonical Game
                # ----------------------------------------------------------
                "game_uuid": game_uuid,
            },
        }

        payloads.append(payload)

    return payloads


# ------------------------------------------------------------------
# Local test harness
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Mock cleaned RAWG data (minimal)
    mock_cleaned_data = {
        "game_id": 12345,
        "platforms": [
            {
                "platform_name": "PC",
                "platform_family": "PC",
                "release_date": "2023-09-01",
                "requirements_minimum": "Intel i5, 8GB RAM",
                "requirements_recommended": "Intel i7, 16GB RAM",
            },
            {
                "platform_name": "PlayStation 5",
                "platform_family": "PlayStation",
                "release_date": "2023-09-01",
                "requirements_minimum": None,
                "requirements_recommended": None,
            },
        ],
    }

    dummy_game_uuid = "11111111-2222-3333-4444-555555555555"

    payloads = generate_platform_payloads(
        cleaned_data=mock_cleaned_data,
        game_uuid=dummy_game_uuid,
    )

    print("=== Generated PlatformSpec Payloads ===")
    for p in payloads:
        print(p)
