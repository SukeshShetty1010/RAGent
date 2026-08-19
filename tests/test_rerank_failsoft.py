"""
tests/test_rerank_failsoft.py

Covers what happens when the reranker backend fails — previously
untested, and the behavior that decides whether a reranker outage
degrades quietly or turns into a refusal storm.

Two distinct paths:
  * RAGRetriever._rerank        — local corpus chunks: keep RRF order,
                                  leave `rerank_score` unset.
  * RAGRetriever.score_relevance — web evidence: return None per item,
                                  never a numeric sentinel.

The sentinel distinction is the whole point. `0.0` was harmless against
the local cross-encoder's raw-logit floors, but sits below any sane
refuse floor on Voyage's normalized 0..1 scale — so a Voyage outage
would have refused every web-augmented query instead of degrading.

Both tests patch retriever.rag_retriever._rerank_scores, so neither the
ONNX model nor the network is exercised. RAGRetriever.__init__ opens a
Qdrant client, so the instance is built with __new__ instead.
"""

import pytest

from agent.task_router import TaskType
from retriever.quality_gate import RetrievalQualityGate, QualityStatus
from retriever.corpus_index import CorpusEntityIndex

pytestmark = pytest.mark.unit


def _retriever():
    from retriever.rag_retriever import RAGRetriever

    # Bypass __init__: it connects to Qdrant, which these tests neither
    # need nor should require.
    r = RAGRetriever.__new__(RAGRetriever)
    r.client = None
    return r


def _boom(*args, **kwargs):
    raise RuntimeError("reranker down")


def _short(query, contents):
    """Simulates a reranker backend that silently drops a document --
    the §20 hazard: one score fewer than documents requested."""
    return [0.5] * max(0, len(contents) - 1)


class _StubShortReranker:
    """Stands in for the local cross-encoder's own object, returning one
    fewer score than requested -- exactly what `local` does not itself
    guard against. Used to exercise _rerank_scores' own length check
    (Change 3a) rather than bypassing it by monkeypatching
    _rerank_scores wholesale, as `_short` above does for _rerank's
    independent inner check (Change 3b)."""

    def rerank(self, query, contents, batch_size=1):
        return [0.5] * max(0, len(contents) - 1)


def test_rerank_failure_preserves_rrf_order_and_omits_score(monkeypatch):
    from retriever import rag_retriever

    monkeypatch.setattr(rag_retriever, "_rerank_scores", _boom)

    candidates = [
        {"content": "first", "score": 0.9},
        {"content": "second", "score": 0.8},
        {"content": "third", "score": 0.7},
    ]
    results = _retriever()._rerank("q", candidates, limit=2)

    assert [c["content"] for c in results] == ["first", "second"]
    assert all("rerank_score" not in c for c in results)


def test_score_relevance_failure_returns_none_not_zero(monkeypatch):
    from retriever import rag_retriever

    monkeypatch.setattr(rag_retriever, "_rerank_scores", _boom)

    scores = _retriever().score_relevance("q", ["a", "b"])
    assert scores == [None, None]


def test_score_relevance_empty_contents_short_circuits(monkeypatch):
    from retriever import rag_retriever

    monkeypatch.setattr(rag_retriever, "_rerank_scores", _boom)
    assert _retriever().score_relevance("q", []) == []


def test_rerank_short_score_list_preserves_rrf_order_and_omits_score(monkeypatch):
    """§20 regression: a reranker backend that returns fewer scores than
    documents (the `local` provider has no such guard) must not leave a
    half-scored, unsortable list. _rerank_scores' length check turns
    this into the same fail-soft path as a hard failure."""
    from retriever import rag_retriever

    monkeypatch.setattr(rag_retriever, "_rerank_scores", _short)

    candidates = [
        {"content": "first", "score": 0.9},
        {"content": "second", "score": 0.8},
        {"content": "third", "score": 0.7},
    ]
    results = _retriever()._rerank("q", candidates, limit=3)

    assert [c["content"] for c in results] == ["first", "second", "third"]
    assert all("rerank_score" not in c for c in results)


def test_rerank_scores_raises_on_short_provider_result(monkeypatch):
    """Change 3a's guard itself: _rerank_scores must not let a
    short-by-one provider result (what `local` returns unguarded)
    through as if it were valid."""
    from retriever import rag_retriever

    monkeypatch.setattr(rag_retriever, "RERANKER_PROVIDER", "local")
    monkeypatch.setattr(rag_retriever, "reranker", _StubShortReranker())

    with pytest.raises(ValueError):
        rag_retriever._rerank_scores("q", ["a", "b", "c"])


def test_score_relevance_short_provider_result_returns_all_none(monkeypatch):
    from retriever import rag_retriever

    monkeypatch.setattr(rag_retriever, "RERANKER_PROVIDER", "local")
    monkeypatch.setattr(rag_retriever, "reranker", _StubShortReranker())

    scores = _retriever().score_relevance("q", ["a", "b", "c"])
    assert scores == [None, None, None]


def test_gate_does_not_refuse_when_rerank_scores_are_absent(monkeypatch):
    """End of the chain: with no rerank_score on any chunk the gate must
    skip the relevance ladder, not refuse."""
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    gate = RetrievalQualityGate(entity_index=CorpusEntityIndex.from_titles(["Far Cry 5"]))

    report = gate.evaluate(
        query="What platforms can I play Far Cry 5 on?",
        task=TaskType.FACTUAL,
        chunks=[{"source_title": "Far Cry 5 Wiki", "content": "Editorial content.", "score": 0.9}],
    )
    assert report.status == QualityStatus.QUALITY_OK
    assert report.max_relevance is None
