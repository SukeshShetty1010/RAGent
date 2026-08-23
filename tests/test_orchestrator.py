"""
tests/test_orchestrator.py

Hermetic tests for retriever/orchestrator.py's post-web-merge re-gate
(STEP 5 in RetrievalOrchestrator.run()). This path had zero coverage
before T25: it concatenates local_chunks + web_chunks and re-runs
RetrievalQualityGate.evaluate() over the merge, which is exactly what
let a single high-scoring web snippet overturn a justified refusal
(see AUDIT_TASKS.md T25 / quality_gate.py's source-scoped ceiling).

RetrievalOrchestrator.__init__ constructs a real RAGRetriever(), which
pulls in the fastembed ONNX reranker — not hermetic. These tests build
the instance via __new__ and inject stub retriever/web_tool instead.
"""

import pytest

from agent.task_router import RouterDecision, TaskType
from retriever.corpus_index import CorpusEntityIndex
from retriever.orchestrator import RetrievalOrchestrator
from retriever.quality_gate import RetrievalQualityGate, QualityStatus
from retriever.strategy_selector import RetrievalConfiguration

pytestmark = pytest.mark.unit


class _StubRetriever:
    def __init__(self, local_chunks, web_scores):
        self._local_chunks = local_chunks
        self._web_scores = web_scores

    def retrieve(self, query, limit):
        return list(self._local_chunks)

    def score_relevance(self, query, contents):
        return list(self._web_scores[: len(contents)])


class _StubWebTool:
    def __init__(self, web_chunks):
        self._web_chunks = web_chunks

    def search(self, query, max_results):
        return list(self._web_chunks[:max_results])


def _make_orchestrator(local_chunks, web_chunks, web_scores, known_titles=("Far Cry 5",)):
    orch = RetrievalOrchestrator.__new__(RetrievalOrchestrator)
    orch.retriever = _StubRetriever(local_chunks, web_scores)
    orch.quality_gate = RetrievalQualityGate(
        entity_index=CorpusEntityIndex.from_titles(list(known_titles))
    )
    orch.web_tool = _StubWebTool(web_chunks)
    return orch


def _config(**overrides):
    defaults = dict(
        limit=5,
        use_window_expansion=False,
        use_query_decomposition=False,
        allow_web_fallback=True,
    )
    defaults.update(overrides)
    return RetrievalConfiguration(**defaults)


def _decision(task=TaskType.FACTUAL):
    return RouterDecision(task=task, intent_signals=set(), reason="test")


def test_strong_web_evidence_cannot_promote_empty_corpus_to_ok(monkeypatch):
    """
    Golden-set g049 shape: corpus evidence scores below the refuse
    floor (QUALITY_EMPTY pre-web triggers the hard, no-LLM web-search
    path), a web result then scores very high. Post-merge status must
    not become QUALITY_OK, and pre_web_quality_report must record the
    original EMPTY verdict.
    """
    monkeypatch.setenv("RERANKER_PROVIDER", "local")

    local_chunks = [
        {
            "source_title": "Far Cry 5 Wiki",
            "content": "Some low-relevance corpus content.",
            "score": 0.5,
            "rerank_score": -10.0,
        }
    ]
    web_chunks = [
        {
            "source_title": "Far Cry 5 Platforms — GameHub",
            "content": "Far Cry 5 is available on PS4, Xbox One, and PC.",
            "score": 0.9,
            "source_url": "https://gamehub.example.com/far-cry-5",
            "source_type": "web",
        }
    ]

    orch = _make_orchestrator(local_chunks, web_chunks, web_scores=[10.0])

    chunks, merge_state, quality_report, web_decision, pre_web_quality_report = orch.run(
        query="What platforms can I play Far Cry 5 on?",
        decision=_decision(),
        config=_config(),
    )

    assert pre_web_quality_report.status == QualityStatus.QUALITY_EMPTY
    assert merge_state == "LOCAL_PLUS_WEB"
    assert quality_report.status != QualityStatus.QUALITY_OK
    assert quality_report.web_max_relevance == 10.0
    assert quality_report.corpus_max_relevance == -10.0


def test_no_web_contribution_leaves_report_untouched(monkeypatch):
    monkeypatch.setenv("RERANKER_PROVIDER", "local")

    local_chunks = [
        {
            "source_title": "Far Cry 5 Wiki",
            "content": "Solid corpus content about the game.",
            "score": 0.9,
            "rerank_score": 6.0,
        }
    ]

    orch = _make_orchestrator(local_chunks, web_chunks=[], web_scores=[])

    chunks, merge_state, quality_report, web_decision, pre_web_quality_report = orch.run(
        query="What platforms can I play Far Cry 5 on?",
        decision=_decision(),
        config=_config(),
    )

    assert quality_report is pre_web_quality_report
    assert quality_report.status == QualityStatus.QUALITY_OK
    assert merge_state == "LOCAL_ONLY"
