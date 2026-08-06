# ============================================================
# retriever/quality_gate.py
# Step 4: Retrieval Quality Gate (Signal-Based, FINAL)
# ============================================================

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Set

from agent.task_router import TaskType


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

    NOISE_KEYWORDS: Set[str] = {
        "sale", "sales", "discount", "deal", "bundle", "price", "store",
        "buy", "purchase",
        "community", "forum", "thread", "discussion",
    }

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
            )
            self._log(report)
            return report

        valid_chunks: List[Dict[str, Any]] = []
        scores: List[float] = []
        temporal_signal = False

        for c in chunks:
            title = (c.get("source_title") or "").lower()
            content = (c.get("content") or "").lower()
            score = float(c.get("score", 0.0))

            if self.is_noise(title, content):
                continue

            valid_chunks.append(c)
            scores.append(score)

            if self._has_temporal_signal(title, content):
                temporal_signal = True

        if not valid_chunks:
            report = QualityReport(
                status=QualityStatus.QUALITY_WEAK,
                reason="Only noise content detected",
                confidence_score=0.0,
            )
            self._log(report)
            return report

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Task-specific strictness
        if task == TaskType.FACTUAL and avg_score < 0.5:
            report = QualityReport(
                status=QualityStatus.QUALITY_WEAK,
                reason="Low semantic similarity for factual query",
                confidence_score=avg_score,
                has_temporal_signal=temporal_signal,
            )
            self._log(report)
            return report

        report = QualityReport(
            status=QualityStatus.QUALITY_OK,
            reason="Evidence sufficient for LLM reasoning",
            confidence_score=avg_score,
            has_temporal_signal=temporal_signal,
        )
        self._log(report)
        return report

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def is_noise(self, title: str, content: str) -> bool:
        text = f"{title} {content}".lower()
        return any(k in text for k in self.NOISE_KEYWORDS)

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
