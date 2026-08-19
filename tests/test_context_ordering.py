"""
tests/test_context_ordering.py

Covers the T2/T20 fix: context ordering must rank by the
cross-encoder's `rerank_score` when every chunk has one, and fall back
to the RRF/retrieval `score` only when the list is not fully reranked
-- never mix the two scales within one sort.

Every test below builds `score` and `rerank_score` to disagree, so a
passing test can only mean the right field won.
"""

import pytest

from agent.context_algorithms import (
    is_fully_reranked,
    relevance_key,
    order_comparison,
    order_listicle,
    order_factual,
)
from agent.context_assembler import ContextAssembler
from agent.task_router import TaskType

pytestmark = pytest.mark.unit


def _chunk(content, score, rerank_score=None, **extra):
    c = {"content": content, "score": score, **extra}
    if rerank_score is not None:
        c["rerank_score"] = rerank_score
    return c


def test_order_factual_ranks_by_rerank_score_when_fully_reranked():
    low_score_high_rerank = _chunk("a", score=0.01, rerank_score=9.0)
    high_score_low_rerank = _chunk("b", score=0.03, rerank_score=-5.0)

    result = order_factual([high_score_low_rerank, low_score_high_rerank])

    assert [c["content"] for c in result] == ["a", "b"]


def test_order_factual_falls_back_to_score_when_no_rerank_score():
    low_score = _chunk("a", score=0.01)
    high_score = _chunk("b", score=0.03)

    result = order_factual([low_score, high_score])

    assert [c["content"] for c in result] == ["b", "a"]


def test_order_factual_mixed_list_falls_back_to_score_for_whole_list():
    """The §2/§20 coupling: a partially-scored list (one reranker call
    failed) must not sort by whichever field each chunk happens to
    carry -- it must fall back to `score` for every chunk."""
    reranked_but_low_score = _chunk("a", score=0.01, rerank_score=9.0)
    unreranked_but_high_score = _chunk("b", score=0.03)

    result = order_factual([reranked_but_low_score, unreranked_but_high_score])

    assert [c["content"] for c in result] == ["b", "a"]


def test_order_comparison_picks_top_entity_chunk_by_rerank_score():
    e1_low = _chunk("e1-low", score=0.05, rerank_score=-1.0, retrieval_context="Entity1")
    e1_high = _chunk("e1-high", score=0.01, rerank_score=8.0, retrieval_context="Entity1")
    e2_low = _chunk("e2-low", score=0.05, rerank_score=-2.0, retrieval_context="Entity2")
    e2_high = _chunk("e2-high", score=0.01, rerank_score=7.0, retrieval_context="Entity2")

    result = order_comparison([e1_low, e1_high, e2_low, e2_high])

    top_two = {c["content"] for c in result[:2]}
    assert top_two == {"e1-high", "e2-high"}


def test_order_listicle_preserves_chunk_index_ranks_tail_by_rerank_score():
    first = _chunk("first", score=0.01, chunk_index=0)
    second = _chunk("second", score=0.01, chunk_index=1)
    tail_low_score_high_rerank = _chunk("tail-a", score=0.01, rerank_score=9.0)
    tail_high_score_low_rerank = _chunk("tail-b", score=0.05, rerank_score=-3.0)

    result = order_listicle(
        [tail_high_score_low_rerank, second, tail_low_score_high_rerank, first]
    )

    assert [c["content"] for c in result] == ["first", "second", "tail-a", "tail-b"]


def test_is_fully_reranked_empty_list_is_false():
    assert is_fully_reranked([]) is False


def test_relevance_key_tolerates_none_score():
    c = _chunk("a", score=None)
    key = relevance_key([c])
    assert key(c) == 0.0


def test_context_assembler_end_to_end_prefers_reranked_chunk():
    """With realistic content sizes the 4000-char budget admits only a
    couple of chunks -- the cross-encoder's #1 pick must survive and
    the RRF-favoured decoy must not."""
    filler = "x" * 1500

    reranker_top_pick = _chunk(
        f"BEST {filler}", score=0.01, rerank_score=9.5, source_title="Doc A"
    )
    rrf_favoured_decoy = _chunk(
        f"DECOY {filler}", score=0.03, rerank_score=-6.0, source_title="Doc B"
    )
    third_chunk = _chunk(
        f"THIRD {filler}", score=0.02, rerank_score=1.0, source_title="Doc C"
    )

    result = ContextAssembler().assemble(
        [rrf_favoured_decoy, reranker_top_pick, third_chunk], TaskType.FACTUAL
    )

    contents = [c["content"] for c in result]
    assert any(c.startswith("BEST") for c in contents)
    assert not any(c.startswith("DECOY") for c in contents)
