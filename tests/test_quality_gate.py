"""
tests/test_quality_gate.py

Hermetic tests for retriever/quality_gate.py's RetrievalQualityGate.
Zero coverage existed for this module before Phase 3.5 — it is the
file most directly responsible for the honesty gate never firing
(see flagship.md Phase 3.5). The entity index is injected via
CorpusEntityIndex.from_titles() so no Qdrant/network access is needed.
"""

import pytest

from agent.task_router import TaskType
from retriever.corpus_index import CorpusEntityIndex
from retriever.quality_gate import RetrievalQualityGate, QualityStatus

pytestmark = pytest.mark.unit


@pytest.fixture
def gate() -> RetrievalQualityGate:
    entity_index = CorpusEntityIndex.from_titles(["Far Cry 5"])
    return RetrievalQualityGate(entity_index=entity_index)


def _chunk(content="Some editorial content about the game.", title="Far Cry 5 Wiki", rerank_score=None, score=0.8):
    c = {"source_title": title, "content": content, "score": score}
    if rerank_score is not None:
        c["rerank_score"] = rerank_score
    return c


def test_empty_chunk_list_is_empty(gate):
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=[])
    assert report.status == QualityStatus.QUALITY_EMPTY
    assert report.evidence_count == 0


def test_below_refuse_floor_is_empty(gate):
    chunks = [_chunk(rerank_score=-6.0)]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_EMPTY
    assert report.max_relevance == -6.0


def test_between_floors_is_weak(gate):
    chunks = [_chunk(rerank_score=0.0)]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_WEAK


def test_above_weak_floor_is_ok(gate):
    chunks = [_chunk(rerank_score=5.0)]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_OK
    assert report.max_relevance == 5.0


def test_missing_rerank_score_skips_floor_never_refuses(gate):
    """
    Reranker down / ablation mode: no chunk carries rerank_score. The
    relevance ladder must be skipped entirely rather than treated as
    zero relevance — a degraded reranker must not become a refusal
    storm.
    """
    chunks = [_chunk(rerank_score=None, score=0.9)]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status != QualityStatus.QUALITY_EMPTY
    assert report.max_relevance is None


def test_entity_ungrounded_query_is_empty_regardless_of_relevance(gate):
    """
    High relevance score is not enough — a query naming a game absent
    from the corpus must refuse, e.g. same-franchise plot chunks
    scoring well against the wrong entry (GTA VI pulling GTA V chunks).
    """
    chunks = [_chunk(title="Far Cry 5 Wiki", content="Unrelated content.", rerank_score=5.5)]
    report = gate.evaluate(
        query="What is the plot of Cyberpunk 2077?",
        task=TaskType.FACTUAL,
        chunks=chunks,
    )
    assert report.status == QualityStatus.QUALITY_EMPTY
    assert report.entity_grounded is False


def test_entity_grounded_query_uses_relevance_ladder(gate):
    chunks = [_chunk(title="Far Cry 5 Wiki", rerank_score=5.0)]
    report = gate.evaluate(
        query="What platforms can I play Far Cry 5 on?",
        task=TaskType.FACTUAL,
        chunks=chunks,
    )
    assert report.entity_grounded is True
    assert report.status == QualityStatus.QUALITY_OK


def test_query_with_no_entity_span_skips_entity_check(gate):
    chunks = [_chunk(title="General Gaming News", rerank_score=5.0)]
    report = gate.evaluate(
        query="What is the release date?",
        task=TaskType.FACTUAL,
        chunks=chunks,
    )
    assert report.entity_grounded is None
    assert report.status == QualityStatus.QUALITY_OK


def test_is_noise_uses_word_boundaries_not_substrings(gate):
    """
    Regression: 'ideal' must not trip the 'deal' noise keyword via
    substring containment (nor 'wholesale' trip 'sale', nor 'priced'
    trip 'price').
    """
    chunks = [_chunk(content="This build feels ideal for speedrunners.", rerank_score=5.0)]
    report = gate.evaluate(
        query="What is a good build for this game?",
        task=TaskType.FACTUAL,
        chunks=chunks,
    )
    assert report.evidence_count == 1
    assert report.status == QualityStatus.QUALITY_OK


def test_is_noise_still_catches_real_noise_keywords(gate):
    chunks = [_chunk(content="Check out this weekend's store sale and discount bundle.", rerank_score=5.0)]
    report = gate.evaluate(
        query="What is a good build for this game?",
        task=TaskType.FACTUAL,
        chunks=chunks,
    )
    assert report.evidence_count == 0
    assert report.status == QualityStatus.QUALITY_WEAK
    assert report.reason == "Only noise content detected"
