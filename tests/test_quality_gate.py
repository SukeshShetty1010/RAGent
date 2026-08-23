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
from utils.observability import MetricsRegistry

pytestmark = pytest.mark.unit


@pytest.fixture
def gate(monkeypatch) -> RetrievalQualityGate:
    # Floors are provider-scoped; pin the provider so these tests measure
    # the ladder rather than whatever RERANKER_PROVIDER happens to be set
    # to in the developer's environment.
    monkeypatch.setenv("RERANKER_PROVIDER", "local")
    entity_index = CorpusEntityIndex.from_titles(["Far Cry 5"])
    return RetrievalQualityGate(entity_index=entity_index)


def _floors(gate: RetrievalQualityGate):
    floors = gate._resolve_floors()
    assert floors is not None, "local provider must have calibrated floors"
    return floors


def _chunk(content="Some editorial content about the game.", title="Far Cry 5 Wiki", rerank_score=None, score=0.8, url=None, source_type=None):
    c = {"source_title": title, "content": content, "score": score}
    if rerank_score is not None:
        c["rerank_score"] = rerank_score
    if url is not None:
        c["source_url"] = url
    if source_type is not None:
        c["source_type"] = source_type
    return c


def test_empty_chunk_list_is_empty(gate):
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=[])
    assert report.status == QualityStatus.QUALITY_EMPTY
    assert report.evidence_count == 0


def test_below_refuse_floor_is_empty(gate):
    refuse_floor, _ = _floors(gate)
    score = refuse_floor - 3.0
    chunks = [_chunk(rerank_score=score)]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_EMPTY
    assert report.max_relevance == score


def test_between_floors_is_weak(gate):
    refuse_floor, weak_floor = _floors(gate)
    chunks = [_chunk(rerank_score=(refuse_floor + weak_floor) / 2)]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_WEAK


def test_above_weak_floor_is_ok(gate):
    _, weak_floor = _floors(gate)
    score = weak_floor + 3.0
    chunks = [_chunk(rerank_score=score)]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_OK
    assert report.max_relevance == score


def test_uncalibrated_provider_skips_floor_never_refuses(gate, monkeypatch):
    """
    A reranker provider with no calibrated floors (_FLOORS[...] is None)
    must skip the relevance ladder entirely. Thresholding a foreign
    score scale is the failure mode this guards: Voyage's normalized
    0..1 scores are all below the local REFUSE_FLOOR of -3.0, so
    applying local floors to them would refuse every single query.
    """
    monkeypatch.setitem(RetrievalQualityGate._FLOORS, "voyage", None)
    monkeypatch.setenv("RERANKER_PROVIDER", "voyage")

    chunks = [_chunk(rerank_score=0.42)]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)

    assert report.status == QualityStatus.QUALITY_OK
    assert report.max_relevance is None
    assert "relevance floor skipped" in report.reason


def test_calibrated_provider_uses_its_own_floors(gate, monkeypatch):
    """Once a provider has floors, the ladder applies on that provider's
    own scale — a 0.05 on a 0..1 scale refuses, where the same number
    would land WEAK under the local logit floors."""
    monkeypatch.setitem(RetrievalQualityGate._FLOORS, "voyage", (0.1, 0.5))
    monkeypatch.setenv("RERANKER_PROVIDER", "voyage")

    report = gate.evaluate(
        query="What platforms can I play Far Cry 5 on?",
        task=TaskType.FACTUAL,
        chunks=[_chunk(rerank_score=0.05)],
    )
    assert report.status == QualityStatus.QUALITY_EMPTY

    report = gate.evaluate(
        query="What platforms can I play Far Cry 5 on?",
        task=TaskType.FACTUAL,
        chunks=[_chunk(rerank_score=0.8)],
    )
    assert report.status == QualityStatus.QUALITY_OK


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


# --------------------------------------------------------------------
# T18: is_noise() must not discard ordinary review prose that merely
# mentions a commerce/forum word in passing. Only the SOURCE (title,
# URL) or a dense cluster of such words in the body should count.
# --------------------------------------------------------------------

PROSE_EXAMPLES = [
    "The open world gives players a great deal of freedom to explore.",
    "You can browse cosmetics in the in-game store without spending real money.",
    "The modding community has kept this game alive for years.",
    "Death carries a steep price of failure in the late game.",
    "There is a lengthy discussion between the two characters about loyalty.",
    "Before the final battle, the player must purchase upgrades from the outpost.",
]


@pytest.mark.parametrize("content", PROSE_EXAMPLES)
def test_is_noise_survives_ordinary_review_prose(gate, content):
    chunks = [_chunk(content=content, title="Far Cry 5 Review", rerank_score=5.0)]
    report = gate.evaluate(
        query="What is a good build for this game?",
        task=TaskType.FACTUAL,
        chunks=chunks,
    )
    assert report.evidence_count == 1
    assert report.status != QualityStatus.QUALITY_WEAK or report.reason != "Only noise content detected"


def test_noise_prose_does_not_break_entity_grounding():
    """
    The §8-interaction regression: a chunk titled after the queried
    entity, whose body incidentally mentions a single commerce word,
    must still survive into assess_grounding() so the query grounds.
    Before the T18 fix this chunk was dropped, source_titles lost
    "far cry 5", and the query was falsely refused as QUALITY_EMPTY.
    """
    entity_index = CorpusEntityIndex.from_titles(["Far Cry 5"])
    gate = RetrievalQualityGate(entity_index=entity_index)

    chunks = [
        _chunk(
            title="Far Cry 5 Review",
            content="Combat rewards a great deal of freedom in how you approach outposts.",
            rerank_score=5.0,
        )
    ]
    report = gate.evaluate(
        query="Far Cry 5 combat",
        task=TaskType.FACTUAL,
        chunks=chunks,
    )
    assert report.entity_grounded is not False
    assert report.status != QualityStatus.QUALITY_EMPTY


def test_storefront_title_is_still_noise(gate):
    chunks = [_chunk(title="Steam Summer Sale", content="Ordinary editorial text.", rerank_score=5.0)]
    report = gate.evaluate(
        query="What is a good build for this game?",
        task=TaskType.FACTUAL,
        chunks=chunks,
    )
    assert report.evidence_count == 0


def test_commerce_url_with_clean_prose_is_still_noise(gate):
    chunks = [
        _chunk(
            title="Far Cry 5",
            content="Ordinary editorial text with no flagged words at all.",
            url="https://example.com/store/far-cry-5",
            rerank_score=5.0,
        )
    ]
    report = gate.evaluate(
        query="What is a good build for this game?",
        task=TaskType.FACTUAL,
        chunks=chunks,
    )
    assert report.evidence_count == 0


def test_chunks_dropped_as_noise_metric_increments(gate):
    MetricsRegistry.get().reset()
    chunks = [
        _chunk(content="Check out this weekend's store sale and discount bundle.", rerank_score=5.0),
        _chunk(content="Ordinary editorial prose about the story.", rerank_score=5.0),
    ]
    gate.evaluate(query="What is a good build for this game?", task=TaskType.FACTUAL, chunks=chunks)
    counters = MetricsRegistry.get().generate_report()["counters"]
    assert counters.get("chunks_dropped_as_noise") == 1


# --------------------------------------------------------------------
# T25: the post-web-merge re-gate must not let web evidence promote a
# verdict the corpus evidence alone did not earn. See orchestrator.py's
# STEP 5 (ChunkMerge/QualityGateReMerge) — this is what evaluate() sees
# on that second call, merged local_chunks + web_chunks in one list.
# --------------------------------------------------------------------

def test_web_score_alone_cannot_lift_empty_to_ok(gate):
    refuse_floor, weak_floor = _floors(gate)
    chunks = [
        _chunk(title="Far Cry 5 Wiki", rerank_score=refuse_floor - 3.0),
        _chunk(title="Some Web Page", rerank_score=weak_floor + 5.0, source_type="web"),
    ]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_EMPTY
    assert report.corpus_max_relevance == refuse_floor - 3.0
    assert report.web_max_relevance == weak_floor + 5.0


def test_web_score_alone_cannot_lift_weak_to_ok(gate):
    refuse_floor, weak_floor = _floors(gate)
    chunks = [
        _chunk(title="Far Cry 5 Wiki", rerank_score=(refuse_floor + weak_floor) / 2),
        _chunk(title="Some Web Page", rerank_score=weak_floor + 5.0, source_type="web"),
    ]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_WEAK


def test_corpus_already_ok_stays_ok_with_web_present(gate):
    refuse_floor, weak_floor = _floors(gate)
    chunks = [
        _chunk(title="Far Cry 5 Wiki", rerank_score=weak_floor + 3.0),
        _chunk(title="Some Web Page", rerank_score=weak_floor + 5.0, source_type="web"),
    ]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_OK


def test_pure_web_evidence_is_empty(gate):
    _, weak_floor = _floors(gate)
    chunks = [
        _chunk(title="Some Web Page", rerank_score=weak_floor + 5.0, source_type="web"),
    ]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_EMPTY
    assert report.corpus_max_relevance is None


def test_missing_rerank_score_skip_path_wins_over_ceiling(gate):
    """
    Reranker unavailable on this call (no chunk anywhere carries
    rerank_score, corpus or web) must still hit the existing fail-soft
    OK skip path — the new ceiling logic must never run ahead of it.
    """
    chunks = [
        _chunk(title="Far Cry 5 Wiki", rerank_score=None),
        _chunk(title="Some Web Page", rerank_score=None, source_type="web"),
    ]
    report = gate.evaluate(query="What platforms can I play Far Cry 5 on?", task=TaskType.FACTUAL, chunks=chunks)
    assert report.status == QualityStatus.QUALITY_OK
    assert "relevance floor skipped" in report.reason
