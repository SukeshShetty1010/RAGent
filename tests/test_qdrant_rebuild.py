"""
tests/test_qdrant_rebuild.py

Local verification tests for the Qdrant cloud rebuild.
No live API calls — reads .env and bulk_ingest source only.
"""

import os
import ast
import pathlib
import pytest
from dotenv import load_dotenv

load_dotenv()


def _get_top_games_list() -> list:
    """Extract TOP_100_GAMES list from bulk_ingest.py via AST parsing.

    Handles both plain assignment (ast.Assign) and annotated assignment
    (ast.AnnAssign) since the variable is declared as:
        TOP_100_GAMES: List[str] = [...]
    """
    src = pathlib.Path("scripts/bulk_ingest.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    for node in ast.walk(tree):
        # Plain assignment: TOP_100_GAMES = [...]
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "TOP_100_GAMES"
                    and isinstance(node.value, ast.List)
                ):
                    return [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                    ]
        # Annotated assignment: TOP_100_GAMES: List[str] = [...]
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "TOP_100_GAMES"
            and isinstance(node.value, ast.List)
        ):
            return [
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant)
            ]
    return []


@pytest.mark.unit
def test_top_100_games_list_exists():
    """bulk_ingest.py must define TOP_100_GAMES (not TOP_300_GAMES)."""
    src = pathlib.Path("scripts/bulk_ingest.py").read_text(encoding="utf-8")
    assert "TOP_100_GAMES" in src, "TOP_100_GAMES not found in bulk_ingest.py"
    assert "TOP_300_GAMES" not in src, (
        "TOP_300_GAMES still present — rename to TOP_100_GAMES"
    )


@pytest.mark.unit
def test_top_100_games_has_100_entries():
    """TOP_100_GAMES must contain exactly 100 game names."""
    games = _get_top_games_list()
    assert len(games) == 100, f"Expected 100 games, got {len(games)}"


@pytest.mark.unit
def test_top_100_games_no_duplicates():
    """TOP_100_GAMES must not contain duplicate game names."""
    games = _get_top_games_list()
    seen = set()
    duplicates = []
    for g in games:
        if g in seen:
            duplicates.append(g)
        seen.add(g)
    assert not duplicates, f"Duplicate game names found: {duplicates}"


@pytest.mark.live
def test_qdrant_url_points_to_cloud():
    """QDRANT_URL in .env must point to the new cloud cluster, not localhost."""
    url = os.environ.get("QDRANT_URL", "")
    assert url, "QDRANT_URL is not set in .env"
    assert "localhost" not in url, f"QDRANT_URL still points to localhost: {url!r}"
    assert "cloud.qdrant.io" in url, (
        f"QDRANT_URL does not look like a Qdrant cloud endpoint: {url!r}"
    )


@pytest.mark.live
def test_qdrant_api_key_is_set():
    """QDRANT_API_KEY in .env must be non-empty."""
    key = os.environ.get("QDRANT_API_KEY", "")
    assert key, "QDRANT_API_KEY is not set in .env"
    assert len(key) > 20, "QDRANT_API_KEY looks too short to be a valid JWT token"
