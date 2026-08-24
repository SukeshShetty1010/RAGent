"""
tests/test_measure_noise_drop_rate.py

Unit tests for evaluation/measure_noise_drop_rate.py's retired pre-T18
rule reimplementation (_old_rule_hits) and its production-parity wrapper
around the real RetrievalQualityGate.is_noise() (_new_rule_reason) --
synthetic title/content/url only, no live Qdrant scroll.
"""

from __future__ import annotations

import pytest

from evaluation.measure_noise_drop_rate import _old_rule_hits, _new_rule_reason
from retriever.quality_gate import RetrievalQualityGate

pytestmark = pytest.mark.unit


@pytest.fixture
def gate() -> RetrievalQualityGate:
    return RetrievalQualityGate()


def test_old_rule_hits_matches_a_single_incidental_mention():
    """The old rule (pre-T18) flags on any single match, unlike the new
    rule's density requirement for content."""
    keywords = {"sale", "forum"}
    hits = _old_rule_hits("Review", "This game has a great deal of freedom to explore.", keywords)
    assert hits == set()  # "deal" isn't in this keyword set; sanity check

    hits = _old_rule_hits("Review", "Check out our forum for discussion.", keywords)
    assert hits == {"forum"}


def test_old_rule_hits_respects_word_boundaries():
    keywords = {"sale"}
    # "wholesale" contains "sale" as a substring but not as a whole word.
    assert _old_rule_hits("Wholesale Distribution Deal", "", keywords) == set()
    assert _old_rule_hits("Great Sale Event", "", keywords) == {"sale"}


def test_old_rule_hits_ignores_url_by_design():
    """_old_rule_hits has no url parameter -- the pre-T18 rule never
    looked at it, which is exactly the gap T18's source-match check
    (title+url) closed."""
    keywords = {"store"}
    assert _old_rule_hits("Editorial Review", "Great gameplay overall.", keywords) == set()


def test_new_rule_reason_source_match_on_title(gate):
    result = _new_rule_reason(gate, title="Official Store Page", content="", url="")
    assert result["is_noise"] is True
    assert result["reason"] == "source_match"
    assert "store" in result["keywords"]


def test_new_rule_reason_source_match_on_url(gate):
    result = _new_rule_reason(gate, title="Editorial Review", content="Great gameplay.", url="https://example.com/store/item")
    assert result["is_noise"] is True
    assert result["reason"] == "source_match"


def test_new_rule_reason_content_density_requires_three_distinct_hits(gate):
    result = _new_rule_reason(
        gate,
        title="Editorial Review",
        content="You can buy this bundle at a discount price today.",
        url="",
    )
    assert result["is_noise"] is True
    assert result["reason"] == "content_density"
    assert result["keywords"] >= {"buy", "bundle", "discount", "price"}


def test_new_rule_reason_single_incidental_mention_is_not_noise(gate):
    """This is the exact case T18 fixed: 'a great deal of freedom' should
    not trip the filter on a single incidental mention."""
    result = _new_rule_reason(
        gate,
        title="Editorial Review",
        content="This game offers a great deal of freedom to explore.",
        url="",
    )
    assert result["is_noise"] is False
    assert result["reason"] is None
    assert result["keywords"] == set()


def test_old_and_new_rules_diverge_on_single_mention_recovered_case(gate):
    """The exact 'recovered_by_t18' scenario measure_noise_drop_rate.py
    tallies: old rule drops on the single mention, new rule keeps it."""
    title, content, url = "Editorial Review", "A great deal of freedom awaits.", ""

    old_hits = _old_rule_hits(title, content, gate.SOURCE_NOISE_KEYWORDS)
    new_verdict = _new_rule_reason(gate, title, content, url)

    assert bool(old_hits) is True   # old rule: single "deal" mention -> drop
    assert new_verdict["is_noise"] is False  # new rule: keeps it
