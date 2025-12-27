"""
RAWG Canonical Game Identity Validator

This module enforces the canonical Game contract *before*
any database or vector operations occur.
"""

from typing import Any, Dict


def validate_game_identity(game_obj: Dict[str, Any]) -> bool:
    """
    Validate a canonical RAWG Game identity object.

    Rules enforced:
    - game_id must exist and be an integer
    - title must exist and be a non-empty string
    - release_year must be an integer or None (never a string)

    Returns:
        True if valid

    Raises:
        ValueError if validation fails
    """

    if not isinstance(game_obj, dict):
        raise ValueError("Game identity must be a dictionary")

    game_id = game_obj.get("game_id")
    if game_id is None or not isinstance(game_id, int):
        raise ValueError("Invalid or missing game_id (must be int)")

    title = game_obj.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Invalid or missing title")

    release_year = game_obj.get("release_year")
    if release_year is not None and not isinstance(release_year, int):
        raise ValueError("release_year must be an integer or None")

    return True
