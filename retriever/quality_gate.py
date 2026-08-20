# ============================================================
# retriever/quality_gate.py
# Step 4: Retrieval Quality Gate (Signal-Based, FINAL)
# ============================================================

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
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
    # evaluation/calibrate_relevance.py, NOT assumed. See
    # evaluation/results/relevance_calibration_2026-08-12.json:
    #   should_refuse group max_relevance: min=-7.97 max=4.87 mean=-2.38
    #   answerable group max_relevance:    min=-2.74 max=10.77 mean=6.42
    # REFUSE_FLOOR sits in the gap between the answerable group's
    # minimum (-2.74, "Rust") and the lowest genuinely-unanswerable
    # score that isn't already caught by entity grounding (-4.32,
    # g050). One unanswerable query (g047, "Beyond Good and Evil 2",
    # max_relevance=1.07) has a real corpus Game identity despite no
    # editorial content — entity_grounded=True there, and no floor
    # can separate it from legitimately-weak answerable evidence like
    # RimWorld's 0.02 without causing over-refusal, so it lands WEAK
    # (partial answer, not refused). That is the honest bound of a
    # single scalar signal; see flagship.md Phase 3.5 for the accepted
    # miss.
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
    #   "cloudflare" — @cf/baai/bge-reranker-base on Workers AI, which
    #               applies the sigmoid server-side: 0..1, heavily
    #               saturated (measured 0.99990 for a match, 3.7e-05 for
    #               a miss). Same scale-collapse problem as Voyage — the
    #               local floors would refuse everything — and its
    #               saturation means calibration should expect the signal
    #               in the tails rather than a smooth spread. Stays None
    #               until calibrated.
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
        "local": (-3.0, 2.0),    # ms-marco raw logits, calibrated 2026-08-12
        "hfspace": (-3.0, 2.0),  # same model, same scale — shares that calibration
        "cloudflare": None,      # bge-reranker-base logits — needs its own calibration
        "voyage": None,          # 0..1 normalized — needs its own calibration
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
                rerank_scores.append(float(rerank_score))

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

        if max_relevance < refuse_floor:
            status = QualityStatus.QUALITY_EMPTY
            reason = "No evidence above relevance floor"
        elif max_relevance < weak_floor:
            status = QualityStatus.QUALITY_WEAK
            reason = "Evidence below confident-relevance floor"
        else:
            status = QualityStatus.QUALITY_OK
            reason = "Evidence sufficient for LLM reasoning"

        report = QualityReport(
            status=status,
            reason=reason,
            confidence_score=max_relevance,
            has_temporal_signal=temporal_signal,
            max_relevance=max_relevance,
            entity_grounded=entity_grounded,
            evidence_count=len(valid_chunks),
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
