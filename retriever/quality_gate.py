# ============================================================
# retriever/quality_gate.py
# Step 4: Retrieval Quality Gate (Signal-Based, FINAL)
# ============================================================

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple

from agent.task_router import TaskType
from retriever.corpus_index import CorpusEntityIndex, _get_entity_index
from retriever.reranker_provider import resolve_reranker_provider
from utils.observability import MetricsRegistry


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("RAG_QUALITY_GATE")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Enums & Contracts
# ============================================================

class QualityStatus(Enum):
    QUALITY_OK = "quality_ok"
    QUALITY_WEAK = "quality_weak"
    QUALITY_EMPTY = "quality_empty"


# Ordering for the source-scoped ceiling clamp in evaluate(): web
# evidence may confirm or enrich a verdict the corpus already earned,
# never promote it. EMPTY < WEAK < OK.
_STATUS_RANK: Dict[QualityStatus, int] = {
    QualityStatus.QUALITY_EMPTY: 0,
    QualityStatus.QUALITY_WEAK: 1,
    QualityStatus.QUALITY_OK: 2,
}


def _status_for_relevance(
    max_relevance: float, refuse_floor: float, weak_floor: float
) -> QualityStatus:
    if max_relevance < refuse_floor:
        return QualityStatus.QUALITY_EMPTY
    elif max_relevance < weak_floor:
        return QualityStatus.QUALITY_WEAK
    return QualityStatus.QUALITY_OK


@dataclass(frozen=True)
class QualityReport:
    status: QualityStatus
    reason: str
    confidence_score: float
    has_temporal_signal: bool = False
    # Max cross-encoder rerank_score across evidence; None if no chunk
    # carried one (reranker unavailable / ablation mode).
    max_relevance: Optional[float] = None
    # None = query names no entity (relevance floor is the only signal).
    entity_grounded: Optional[bool] = None
    evidence_count: int = 0
    # Same max, but partitioned by source — corpus (non-web) chunks vs
    # web chunks. None if that partition contributed no scored chunks.
    # See T25: the post-web-merge re-gate must never let a web-only
    # score promote a status the corpus evidence alone did not earn.
    corpus_max_relevance: Optional[float] = None
    web_max_relevance: Optional[float] = None


# ============================================================
# Quality Gate
# ============================================================

class RetrievalQualityGate:
    """
    Evaluates retrieved evidence for:
    - Noise
    - Task alignment
    - Temporal signal presence

    IMPORTANT:
    This gate does NOT decide replacement.
    It only emits signals for the orchestrator.
    """

    # --------------------------------------------------------
    # Noise indicators
    # --------------------------------------------------------

    # These keywords were written to catch storefronts and forums — a
    # URL/title-shaped signal, not a prose one. Ordinary review copy
    # legitimately says "a great deal of freedom" or "the modding
    # community" without being commerce/forum noise. Split accordingly:
    #
    # - SOURCE_NOISE_KEYWORDS matches only source_title + source_url.
    #   A page *titled* "Steam Summer Sale", or served from a /store/
    #   path, is a storefront; a review that mentions one is not.
    # - Content (the prose body) only trips noise on a DENSITY signal —
    #   see is_noise() — because a single incidental keyword in running
    #   text is not evidence the chunk itself is a storefront/forum
    #   blob, but several distinct ones together are.
    SOURCE_NOISE_KEYWORDS: Set[str] = {
        "sale", "sales", "discount", "deal", "bundle", "price", "store",
        "buy", "purchase",
        "community", "forum", "thread", "discussion",
    }

    # Minimum number of DISTINCT keyword hits in the content body before
    # it's treated as noise (vs. a single incidental mention in prose).
    CONTENT_NOISE_DENSITY = 3

    # --------------------------------------------------------
    # Temporal signal patterns
    # --------------------------------------------------------

    TEMPORAL_PATTERNS = [
        r"\b20(2[3-9]|[3-9]\d)\b",   # 2023+
        r"\bpatch\b",
        r"\bupdate\b",
        r"\bhotfix\b",
        r"\bchangelog\b",
        r"\brelease notes\b",
    ]

    # --------------------------------------------------------
    # Relevance floor — thresholds on the cross-encoder rerank_score,
    # NOT the RRF fusion score (identical for a perfect match and pure
    # noise, see rag_retriever.py:245).
    #
    # ms-marco-MiniLM-L-6-v2 turns out to apply Identity, not Sigmoid
    # (raw logits, roughly -8..+11 on this corpus) — confirmed by
    # evaluation/calibrate_relevance.py, NOT assumed. Recalibrated
    # 2026-08-23 against the fully-migrated (Gemini-embedded) corpus —
    # see evaluation/results/relevance_calibration_local_2026-08-23.json:
    #   should_refuse group max_relevance: min=-7.97 max=4.87 mean=-2.35
    #   answerable group max_relevance:    min=-2.74 max=10.77 mean=6.41
    # Distribution is within rounding of the pre-migration
    # relevance_calibration_2026-08-12.json run — the migration changed
    # which chunks retrieval surfaces, but not the score range the
    # cross-encoder assigns them — so REFUSE_FLOOR/WEAK_FLOOR are
    # unchanged, now re-validated rather than merely carried forward.
    # REFUSE_FLOOR=-3.0 sits strictly below the answerable group's
    # minimum (-2.74, "Rust"): zero false refusals on the golden set.
    # WEAK_FLOOR=2.0 lands 5/40 (12.5%) of answerable golden queries
    # WEAK and 0 EMPTY — inside the 10-20% target band, sitting in the
    # real (if narrow, ~0.2 wide) gap between 1.94 and 2.15 rather than
    # an arbitrary percentile.
    #
    # T33 (AUDIT_TASKS.md) replaced entity grounding's title fallback
    # from raw substring containment to a token-prefix test, which
    # changes what this floor is actually backstopping: in the
    # 2026-08-23 run, 9 of the 10 should_refuse queries are caught by
    # entity_grounded=False alone (evaluate() short-circuits to
    # QUALITY_EMPTY before the relevance floor is even consulted, see
    # below) — including g047 ("Beyond Good and Evil 2"), which the
    # pre-T33 substring match had falsely grounded via a fragment like
    # "evil 2" matching an unrelated title. That query now correctly
    # reaches QUALITY_EMPTY through entity grounding rather than
    # sliding through as a WEAK floor miss. The relevance floor's
    # remaining job is the one should_refuse query with no entity
    # verdict at all (g049, entity_grounded=None, max_relevance=-7.97,
    # "chocolate chip cookies") — REFUSE_FLOOR is what catches that one.
    #
    # The floors are PROVIDER-SCOPED because the reranker backends do
    # not all share a score scale. What decides the entry is the MODEL,
    # not the transport:
    #   "local"   — the cross-encoder in-process: raw logits (above).
    #   "hfspace" — the SAME model (Xenova/ms-marco-MiniLM-L-6-v2, same
    #               pinned fastembed build) hosted on an HF Space and
    #               called over HTTP. Same scale, so it shares local's
    #               calibrated floors deliberately — moving the model to
    #               a bigger CPU does not change what it scores. If that
    #               Space's model or fastembed version ever diverges
    #               from hf_space/requirements.txt, this entry stops
    #               being valid and must go back to None.
    #   "cloudflare" — @cf/baai/bge-reranker-base on Workers AI, applying
    #               the sigmoid server-side: 0..1. Calibrated 2026-08-21
    #               against the fully-migrated corpus (see §3/§1 in
    #               AUDIT_TASKS.md) via evaluation/calibrate_relevance.py.
    #               See evaluation/results/relevance_calibration_cloudflare_2026-08-21.json:
    #                 should_refuse group max_relevance: min=0.0036 max=0.9019 mean=0.2124
    #                 answerable group max_relevance:    min=0.0249 max=0.9999 mean=0.9328
    #               Contrary to §1's original prediction, the scale is NOT
    #               saturated at the extremes on this corpus — genuine
    #               partial matches populate the 0.3-0.9 middle (e.g. a
    #               real-but-off-topic Game identity scored 0.33).
    #               REFUSE_FLOOR=0.02 sits strictly below the answerable
    #               group's minimum (0.0249, "Rust") — a conservative
    #               choice mirroring "local"'s derivation above: zero
    #               false refusals on the golden set, at the cost of not
    #               catching two unanswerable queries whose scores (0.10,
    #               0.33) sit above it — both land QUALITY_WEAK instead of
    #               refused, which entity grounding does not catch either
    #               (one has a real Game identity — "Beyond Good and Evil
    #               2" — with no editorial content; see quality_gate.py's
    #               entity-grounding note above). WEAK_FLOOR=0.90 sits in
    #               a genuine gap in the answerable distribution (next
    #               values: 0.8897 below, 0.9601 above), landing exactly
    #               8/40 (20%) of answerable golden queries WEAK and 0
    #               EMPTY — the top of the 10-20% band targeted when this
    #               was calibrated, chosen for sitting on real data
    #               structure rather than an arbitrary percentile.
    #   "voyage"  — a DIFFERENT model emitting normalized 0..1. Applying
    #               the local floors there would make the weak floor of
    #               2.0 unreachable (every query WEAK) and the refuse
    #               floor of -3.0 impossible to trip (the refusal signal
    #               silently dies), so it stays None — "not calibrated
    #               yet" — until evaluation/calibrate_relevance.py has
    #               been re-run against the fully-migrated corpus. None
    #               reuses the existing "no rerank_score" skip path
    #               rather than thresholding an uncalibrated number.
    # --------------------------------------------------------

    _FLOORS: Dict[str, Optional[Tuple[float, float]]] = {
        # provider: (REFUSE_FLOOR, WEAK_FLOOR)
        "local": (-3.0, 2.0),    # ms-marco raw logits, calibrated 2026-08-23
        "hfspace": (-3.0, 2.0),  # same model, same scale — shares that calibration
        "cloudflare": (0.02, 0.90),  # bge-reranker-base 0..1, calibrated 2026-08-21
        "voyage": None,          # 0..1 normalized — needs its own calibration
    }

    # The corpus was fully re-embedded into Gemini's vector space by
    # this date (AUDIT_TASKS.md T1/T3). A calibration artifact produced
    # before it measured a different corpus and no longer describes
    # what the reranker scores today, even if the provider name and
    # floor values happen to look unchanged (see "local" above).
    CORPUS_EMBEDDING_MIGRATION_DATE = date(2026, 8, 21)

    # Which calibration artifact each provider's floors are backed by.
    # Providers sharing a model share an artifact (hfspace -> local) —
    # that is what makes the sharing in _FLOORS valid in the first
    # place. None means "no calibrated floors" (see _FLOORS["voyage"])
    # and has no artifact to point to.
    _CALIBRATION: Dict[str, Optional[str]] = {
        "local": "relevance_calibration_local_2026-08-23.json",
        "hfspace": "relevance_calibration_local_2026-08-23.json",  # same model
        "cloudflare": "relevance_calibration_cloudflare_2026-08-21.json",
        "voyage": None,
    }

    # --------------------------------------------------------

    def __init__(self, entity_index: Optional[CorpusEntityIndex] = None) -> None:
        self._entity_index_override = entity_index

    def _resolve_floors(self) -> Optional[Tuple[float, float]]:
        """(refuse, weak) for the active reranker, or None if that
        provider has no calibrated floors.

        Resolved per call, not cached, so a provider switch (or a test's
        monkeypatch.setenv) takes effect without a module reload. The env
        var is read via retriever/reranker_provider.py rather than
        importing rag_retriever, which would pull the fastembed ONNX
        models into every hermetic quality-gate test.
        """
        return self._FLOORS.get(resolve_reranker_provider())

    def _entity_index(self) -> CorpusEntityIndex:
        if self._entity_index_override is not None:
            return self._entity_index_override
        return _get_entity_index()

    def evaluate(
        self,
        query: str,
        task: TaskType,
        chunks: List[Dict[str, Any]],
    ) -> QualityReport:

        if not chunks:
            report = QualityReport(
                status=QualityStatus.QUALITY_EMPTY,
                reason="No evidence retrieved",
                confidence_score=0.0,
                evidence_count=0,
            )
            self._log(report)
            return report

        valid_chunks: List[Dict[str, Any]] = []
        scores: List[float] = []
        rerank_scores: List[float] = []
        corpus_rerank_scores: List[float] = []
        web_rerank_scores: List[float] = []
        temporal_signal = False

        for c in chunks:
            title = (c.get("source_title") or "").lower()
            content = (c.get("content") or "").lower()
            url = (c.get("source_url") or "").lower()
            score = float(c.get("score", 0.0))

            if self.is_noise(title, content, url):
                MetricsRegistry.get().inc("chunks_dropped_as_noise")
                continue

            valid_chunks.append(c)
            scores.append(score)

            rerank_score = c.get("rerank_score")
            if rerank_score is not None:
                rs = float(rerank_score)
                rerank_scores.append(rs)
                if c.get("source_type") == "web":
                    web_rerank_scores.append(rs)
                else:
                    corpus_rerank_scores.append(rs)

            if self._has_temporal_signal(title, content):
                temporal_signal = True

        if not valid_chunks:
            report = QualityReport(
                status=QualityStatus.QUALITY_WEAK,
                reason="Only noise content detected",
                confidence_score=0.0,
                evidence_count=0,
            )
            self._log(report)
            return report

        entity_grounded = self._entity_index().assess_grounding(query, valid_chunks)

        if entity_grounded is False:
            report = QualityReport(
                status=QualityStatus.QUALITY_EMPTY,
                reason="Query entity absent from corpus",
                confidence_score=0.0,
                has_temporal_signal=temporal_signal,
                entity_grounded=False,
                evidence_count=len(valid_chunks),
            )
            self._log(report)
            return report

        avg_score = sum(scores) / len(scores) if scores else 0.0
        floors = self._resolve_floors()

        if not rerank_scores or floors is None:
            # Either the reranker was unavailable on this call (fail-soft
            # omission in rag_retriever._rerank, or an ablation mode), or
            # the active reranker provider has no calibrated floors. In
            # both cases the relevance ladder is skipped entirely rather
            # than refusing or thresholding an uncalibrated number — a
            # degraded reranker must not turn into a refusal storm.
            cause = (
                "no rerank_score" if not rerank_scores
                else f"floors uncalibrated for provider '{resolve_reranker_provider()}'"
            )
            logger.warning(f"Relevance floor skipped — {cause}")
            report = QualityReport(
                status=QualityStatus.QUALITY_OK,
                reason=f"Evidence present (relevance floor skipped — {cause})",
                confidence_score=avg_score,
                has_temporal_signal=temporal_signal,
                entity_grounded=entity_grounded,
                evidence_count=len(valid_chunks),
            )
            self._log(report)
            return report

        max_relevance = max(rerank_scores)
        refuse_floor, weak_floor = floors

        observed_status = _status_for_relevance(max_relevance, refuse_floor, weak_floor)

        corpus_max_relevance = max(corpus_rerank_scores) if corpus_rerank_scores else None
        web_max_relevance = max(web_rerank_scores) if web_rerank_scores else None

        # Ceiling: what the corpus evidence alone would justify. No
        # corpus-scored chunk at all (pure web, or a merged call whose
        # corpus chunks carried no rerank_score) means the corpus signal
        # is absent, not neutral — ceiling is QUALITY_EMPTY. When there
        # is no web evidence in this call, corpus_max_relevance ==
        # max_relevance and the ceiling trivially equals observed_status,
        # so pre-web-only evaluate() calls are unchanged by construction.
        if corpus_max_relevance is not None:
            ceiling_status = _status_for_relevance(corpus_max_relevance, refuse_floor, weak_floor)
        else:
            ceiling_status = QualityStatus.QUALITY_EMPTY

        status = min(observed_status, ceiling_status, key=lambda s: _STATUS_RANK[s])

        if status is observed_status:
            reason = {
                QualityStatus.QUALITY_EMPTY: "No evidence above relevance floor",
                QualityStatus.QUALITY_WEAK: "Evidence below confident-relevance floor",
                QualityStatus.QUALITY_OK: "Evidence sufficient for LLM reasoning",
            }[status]
        else:
            MetricsRegistry.get().inc("web_relevance_clamped")
            reason = (
                f"Web evidence alone scored {observed_status.name}, but corpus "
                f"evidence only supports {ceiling_status.name} — web cannot "
                f"promote the verdict past what the corpus earned"
            )

        report = QualityReport(
            status=status,
            reason=reason,
            confidence_score=max_relevance,
            has_temporal_signal=temporal_signal,
            max_relevance=max_relevance,
            entity_grounded=entity_grounded,
            evidence_count=len(valid_chunks),
            corpus_max_relevance=corpus_max_relevance,
            web_max_relevance=web_max_relevance,
        )
        self._log(report)
        return report

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def is_noise(self, title: str, content: str, url: str = "") -> bool:
        """A chunk is noise if its SOURCE (title/URL) looks like a
        storefront/forum, or its content body is dense with commerce/
        forum vocabulary (>= CONTENT_NOISE_DENSITY distinct hits) rather
        than mentioning one incidentally in prose.
        """
        source_text = f"{title} {url}".lower()
        if any(
            re.search(rf"\b{re.escape(k)}\b", source_text)
            for k in self.SOURCE_NOISE_KEYWORDS
        ):
            return True

        content_text = content.lower()
        distinct_hits = sum(
            1
            for k in self.SOURCE_NOISE_KEYWORDS
            if re.search(rf"\b{re.escape(k)}\b", content_text)
        )
        return distinct_hits >= self.CONTENT_NOISE_DENSITY

    def _has_temporal_signal(self, title: str, content: str) -> bool:
        text = f"{title} {content}".lower()
        return any(re.search(p, text) for p in self.TEMPORAL_PATTERNS)

    @staticmethod
    def _log(report: QualityReport) -> None:
        logger.info(
            f"🛡️ Quality Gate: {report.status.name} "
            f"(Reason: {report.reason}, "
            f"Confidence: {report.confidence_score:.2f}, "
            f"TemporalSignal={report.has_temporal_signal})"
        )
