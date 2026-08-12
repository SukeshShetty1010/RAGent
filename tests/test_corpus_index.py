"""
tests/test_corpus_index.py

Hermetic tests for retriever/corpus_index.py's CorpusEntityIndex.
Uses CorpusEntityIndex.from_titles() to build the index offline — no
Qdrant, no network.
"""

import pytest

from retriever.corpus_index import CorpusEntityIndex

pytestmark = pytest.mark.unit


@pytest.fixture
def index() -> CorpusEntityIndex:
    return CorpusEntityIndex.from_titles(
        [
            "Far Cry 5",
            "Grand Theft Auto V",
            "Assassin's Creed Valhalla",
        ]
    )


def test_grand_theft_auto_vi_is_not_grand_theft_auto_v(index):
    """
    The headline case this module exists for: "GTA VI" must not be
    considered grounded just because "GTA V" is a substring-adjacent
    corpus title. Token-tuple equality, not substring containment.
    """
    result = index.assess_grounding(
        "What is the release date for Grand Theft Auto VI?",
        evidence=[],
    )
    assert result is False


def test_known_title_is_grounded(index):
    result = index.assess_grounding(
        "What platforms can I play Far Cry 5 on?",
        evidence=[],
    )
    assert result is True


def test_source_title_fallback_grounds_even_if_not_in_known_titles(index):
    """Guards against Game.title drifting from EditorialChunk.source_title."""
    result = index.assess_grounding(
        "What is the plot of Watch Dogs Legion?",
        evidence=[{"source_title": "Watch Dogs Legion Review"}],
    )
    assert result is True


def test_query_with_no_entity_span_is_none(index):
    result = index.assess_grounding("What is the release date?", evidence=[])
    assert result is None


def test_empty_index_never_returns_false(index):
    empty_index = CorpusEntityIndex.from_titles([])
    result = empty_index.assess_grounding(
        "What is the plot of Grand Theft Auto V?",
        evidence=[],
    )
    assert result is None


def test_load_failed_never_returns_false():
    failed_index = CorpusEntityIndex.__new__(CorpusEntityIndex)
    failed_index._load_failed = True
    failed_index.known_titles = set()

    result = failed_index.assess_grounding(
        "What is the plot of Grand Theft Auto V?",
        evidence=[],
    )
    assert result is None


def test_candidate_spans_keeps_digits_and_roman_numerals_whole(index):
    spans = index.candidate_spans("What platforms can I play Far Cry 5 on?")
    assert ("far", "cry", "5") in spans


def test_candidate_spans_ignores_sentence_initial_token(index):
    spans = index.candidate_spans("Grand Theft Auto V release date")
    # The query's first token ("Grand") is never span-seeded, so the
    # entity is missed here by design — callers must phrase queries
    # with the entity mid/end-sentence, which the golden set does.
    assert ("grand", "theft", "auto", "v") not in spans
