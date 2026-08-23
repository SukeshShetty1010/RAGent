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
    # The default (conservative) span set never seeds off the first
    # token, so it misses an entity that opens the sentence. This is
    # the "verdict" set assess_grounding uses to decide whether the
    # query names *any* entity at all — it stays conservative on
    # purpose. See test_sentence_initial_title_is_grounded below for
    # the "match" set, which does include the first token and recovers
    # this exact case at the assess_grounding level.
    assert ("grand", "theft", "auto", "v") not in spans


def test_sentence_initial_title_is_grounded(index):
    """assess_grounding must not miss an entity just because it opens
    the sentence: it searches a wider (greedy) span set than
    candidate_spans()'s default, once the conservative set has
    confirmed the query names something at all."""
    result = index.assess_grounding(
        "Grand Theft Auto V release date",
        evidence=[],
    )
    assert result is True


def test_bare_title_query_is_grounded(index):
    """Regression for AUDIT_TASKS §8: an entity-only query with no
    interrogative lead-in ("What...") must still ground, since the
    corpus does anchor Far Cry 5."""
    result = index.assess_grounding("Far Cry 5 combat", evidence=[])
    assert result is True


def test_leading_verb_query_is_grounded(index):
    result = index.assess_grounding(
        "Compare Far Cry 5 and Doom Eternal", evidence=[]
    )
    assert result is True


def test_leading_non_stopword_without_entity_stays_none(index):
    """A leading non-stopword verb with no known title anywhere in the
    query must not manufacture a refusal."""
    result = index.assess_grounding("Tell me about combat", evidence=[])
    assert result is None


def test_contraction_leading_token_stays_none(index):
    """"What's" is not in _STOPWORDS (the tokenizer keeps apostrophes),
    so a naive "skip index 0 only if it's a stopword" fix would treat
    it as a candidate span and refuse. 5 of the 50 golden-set queries
    open with "What's"."""
    result = index.assess_grounding(
        "What's the best co-op shooter?", evidence=[]
    )
    assert result is None


def test_sentence_initial_unknown_title_still_refuses(index):
    result = index.assess_grounding(
        "Grand Theft Auto VI release date", evidence=[]
    )
    assert result is False


# --------------------------------------------------------------------
# T25 (defect B): the source_title fallback must be a token-PREFIX
# test, not substring containment. A raw substring test let a
# fragmented span match anywhere inside an unrelated corpus title.
# --------------------------------------------------------------------

def test_fragmented_span_does_not_substring_match_unrelated_title(index):
    """
    Regression for golden-set g047: "Beyond Good and Evil 2" splits
    into ('good',) and ('evil', '2') because "and" is deliberately not
    in _CONNECTORS (it separates two different entities in comparison
    queries). Under the old substring rule, "evil 2" matched inside the
    retrieved evidence's "Resident Evil 2" title — an entirely
    different game — falsely grounding the query.
    """
    result = index.assess_grounding(
        "Beyond Good and Evil 2 platforms",
        evidence=[{"source_title": "Resident Evil 2 (2019 video game) — Release"}],
    )
    assert result is False


def test_short_span_does_not_match_mid_word(index):
    """
    Regression for golden-set g050: a one-token span like ('us',) must
    not ground just because "us" appears as a substring inside an
    unrelated title's word (e.g. "Fergus"). Prefix comparison is
    token-based, so "fergus" can never equal "us".
    """
    result = index.assess_grounding(
        "What did the US do in this game?",
        evidence=[{"source_title": "Fergus's Discount Adventures"}],
    )
    assert result is False


def test_title_drift_prefix_match_still_grounds(index):
    """
    The fallback still has to do its actual job: a corpus chunk titled
    "Far Cry 5 Review — GameSpot" (Game.title drifted into an editorial
    headline) must ground a "Far Cry 5" query, since the span is a real
    token-prefix of that title.
    """
    result = index.assess_grounding(
        "What platforms can I play Far Cry 5 on?",
        evidence=[{"source_title": "Far Cry 5 Review — GameSpot"}],
    )
    assert result is True


def test_title_leading_stopword_still_grounds_via_prefix(index):
    """
    Regression: "It Takes Two" and "The Legend of Zelda: ..." open with
    a word candidate_spans() never seeds a span on ("It"/"The" are
    stopwords), so the query's span is missing the title's own first
    token. A naive position-0 prefix test would then never match a
    real, on-topic title — the prefix must be checked after stripping
    the title's own leading stopword run.
    """
    it_takes_two_index = CorpusEntityIndex.from_titles(["It Takes Two"])
    result = it_takes_two_index.assess_grounding(
        "When was It Takes Two released?",
        evidence=[{"source_title": "It Takes Two — Film"}],
    )
    assert result is True


def test_exact_known_title_match_unaffected_by_prefix_change(index):
    single_token_index = CorpusEntityIndex.from_titles(["Control"])
    result = single_token_index.assess_grounding(
        "What is the plot of Control?", evidence=[]
    )
    assert result is True


# --------------------------------------------------------------------
# T34: curly-apostrophe entity grounding regression.
# --------------------------------------------------------------------

def test_curly_apostrophe_query_matches_straight_apostrophe_title(index):
    """
    Regression for AUDIT_TASKS §34 (T34): "Assassin's Creed Valhalla"
    written with a curly right-quote (U+2019, as it appears literally in
    tests/regression_suite.py's BUG-003 case) failed to ground against
    the corpus's real ASCII-apostrophe title, flipping BUG-003 from FULL
    to INSUFFICIENT even though the corpus held real, on-topic Valhalla
    evidence. Confirmed live: the tokenizer split "Assassin's" into
    two tokens ("assassin", "s") instead of one ("assassin's",), which
    never matched the corpus title's token sequence.
    """
    result = index.assess_grounding(
        "Latest update for Assassin’s Creed Valhalla", evidence=[]
    )
    assert result is True


def test_curly_apostrophe_source_title_fallback_still_grounds(index):
    """Same fix, exercised through the source_title prefix-fallback path
    (the branch that actually fires in production, since real
    EditorialChunk source_titles carry the ASCII apostrophe)."""
    result = index.assess_grounding(
        "Latest update for Assassin’s Creed Valhalla",
        evidence=[{"source_title": "Assassin's Creed Valhalla — Gameplay"}],
    )
    assert result is True
