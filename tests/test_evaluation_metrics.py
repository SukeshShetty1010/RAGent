"""
tests/test_evaluation_metrics.py

Hermetic tests for tests/evaluation_metrics.py's entity-matching helpers.
"""

import pytest

from tests.evaluation_metrics import (
    calculate_entity_coverage,
    calculate_evidence_hit_rate,
)

pytestmark = pytest.mark.unit


def test_entity_coverage_matches_across_apostrophe_variants():
    """
    Regression for AUDIT_TASKS §34: RetrievalQualityKPI reported Entity
    Coverage 0.00% for a Valhalla temporal query even though real
    Valhalla chunks were retrieved, because the expected entity string
    used a curly apostrophe (U+2019) while retrieved chunk titles use
    the corpus's real ASCII apostrophe (U+0027) — _entity_match's
    substring test never matched across the two byte sequences.
    """
    retrieved_chunks = [
        {"source_title": "Assassin's Creed Valhalla — Gameplay"},
    ]
    result = calculate_entity_coverage(
        retrieved_chunks, ["Assassin’s Creed Valhalla"]
    )
    assert result.coverage == 1.0
    assert result.covered_entities == 1


def test_evidence_hit_rate_matches_across_apostrophe_variants():
    retrieved_chunks = [
        {"source_title": "Assassin's Creed Valhalla — Gameplay"},
    ]
    result = calculate_evidence_hit_rate(
        retrieved_chunks, ["Assassin’s Creed Valhalla"]
    )
    assert result.hit_rate == 1.0


def test_entity_coverage_still_requires_a_real_match():
    """Sanity check the fix doesn't turn matching into a no-op — an
    unrelated title must still score 0 coverage."""
    retrieved_chunks = [{"source_title": "Far Cry 5 Review"}]
    result = calculate_entity_coverage(
        retrieved_chunks, ["Assassin’s Creed Valhalla"]
    )
    assert result.coverage == 0.0


def test_entity_coverage_falls_through_web_fallback_sentinel_to_source_title():
    """
    Regression for AUDIT_TASKS §34: web-augmented evidence carries
    retrieval_context="fallback" (set by agent/tools/web_search.py as a
    merge-state marker, not an entity name). _resolve_entity's key
    priority checked retrieval_context before source_title, so every
    web-sourced chunk resolved to the literal string "fallback" and
    never matched any expected entity — this is what made
    RetrievalQualityKPI report Entity Coverage 0.00% / Evidence Hit: NO
    for "Latest patch notes for Assassin's Creed Valhalla" (a temporal
    query answered entirely from web evidence) even after the
    apostrophe-normalization fix, confirmed live.
    """
    retrieved_chunks = [
        {
            "source_title": "Assassin's Creed Valhalla Update 1.5.0.1 Patch Notes",
            "game_name": None,
            "canonical_game": None,
            "retrieval_context": "fallback",
        },
    ]
    result = calculate_entity_coverage(
        retrieved_chunks, ["Assassin’s Creed Valhalla"]
    )
    assert result.coverage == 1.0
