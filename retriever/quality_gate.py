# ============================================================
# retriever/quality_gate.py
# Step 4: Retrieval Quality Gate
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Set

from agent.task_router import TaskType


# ============================================================
# Logging Setup
# ============================================================

logger = logging.getLogger("RAG_QUALITY_GATE")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# ENUMS
# ============================================================

class QualityStatus(Enum):
    """
    Strict quality assessment outcomes.
    """
    QUALITY_OK = "quality_ok"
    QUALITY_WEAK = "quality_weak"
    QUALITY_EMPTY = "quality_empty"


# ============================================================
# CONTRACT
# ============================================================

@dataclass(frozen=True)
class QualityReport:
    status: QualityStatus
    reason: str
    confidence_score: float


# ============================================================
# QUALITY GATE
# ============================================================

class RetrievalQualityGate:
    """
    Deterministic post-retrieval validator.

    Decides whether retrieved chunks are sufficient
    to proceed to the LLM or whether a web fallback
    should be triggered.
    """

    # Dataset-specific noise indicators
    NOISE_KEYWORDS: Set[str] = {
        "sale",
        "sales",
        "discount",
        "deal",
        "bundle",
        "price",
        "price drop",
        "store",
        "buy",
        "purchase",
    }

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def evaluate(
        self,
        query: str,
        task: TaskType,
        chunks: List[Dict[str, Any]],
    ) -> QualityReport:
        """
        Evaluate retrieved chunks for quality and alignment.
        """

        # ----------------------------------------------------
        # Step A: Empty Check
        # ----------------------------------------------------
        if not chunks:
            report = QualityReport(
                status=QualityStatus.QUALITY_EMPTY,
                reason="No retrieval evidence found",
                confidence_score=0.0,
            )
            self._log(report)
            return report

        # ----------------------------------------------------
        # Step B: Noise Filtering
        # ----------------------------------------------------
        valid_chunks: List[Dict[str, Any]] = []

        for c in chunks:
            title = (c.get("source_title") or "").lower()
            content = (c.get("content") or "").lower()

            if self._is_noise(title, content):
                continue

            valid_chunks.append(c)

        if not valid_chunks:
            report = QualityReport(
                status=QualityStatus.QUALITY_WEAK,
                reason="Only noise content detected (sales/promotions)",
                confidence_score=0.0,
            )
            self._log(report)
            return report

        # ----------------------------------------------------
        # Step C: Task-Specific Alignment Rules
        # ----------------------------------------------------

        # -------- COMPARISON --------
        if task == TaskType.COMPARISON:
            contexts = {
                c.get("retrieval_context")
                for c in valid_chunks
                if c.get("retrieval_context")
            }

            if len(contexts) < 2:
                report = QualityReport(
                    status=QualityStatus.QUALITY_WEAK,
                    reason="One-sided comparison evidence",
                    confidence_score=self._avg_score(valid_chunks),
                )
                self._log(report)
                return report

        # -------- LISTICLE --------
        if task == TaskType.LISTICLE:
            indices = {
                c.get("chunk_index")
                for c in valid_chunks
                if isinstance(c.get("chunk_index"), int)
            }

            if indices and all(idx == 0 for idx in indices):
                report = QualityReport(
                    status=QualityStatus.QUALITY_WEAK,
                    reason="Shallow listicle content (only headers/intros)",
                    confidence_score=self._avg_score(valid_chunks),
                )
                self._log(report)
                return report

        # -------- FACTUAL --------
        if task == TaskType.FACTUAL:
            top_score = valid_chunks[0].get("score", 0.0)
            if not isinstance(top_score, (int, float)) or top_score < 0.5:
                report = QualityReport(
                    status=QualityStatus.QUALITY_WEAK,
                    reason="Low semantic similarity for factual query",
                    confidence_score=self._avg_score(valid_chunks),
                )
                self._log(report)
                return report

        # ----------------------------------------------------
        # Step D: Default Pass
        # ----------------------------------------------------
        report = QualityReport(
            status=QualityStatus.QUALITY_OK,
            reason="Evidence sufficient for LLM reasoning",
            confidence_score=self._avg_score(valid_chunks),
        )
        self._log(report)
        return report

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _is_noise(self, title: str, content: str) -> bool:
        """
        Detect sales / promotional noise.
        """
        for kw in self.NOISE_KEYWORDS:
            if kw in title:
                return True

        hit_count = sum(1 for kw in self.NOISE_KEYWORDS if kw in content)
        return hit_count >= 2  # dominated by noise terms

    @staticmethod
    def _avg_score(chunks: List[Dict[str, Any]]) -> float:
        scores = [
            c.get("score")
            for c in chunks
            if isinstance(c.get("score"), (int, float))
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _log(report: QualityReport) -> None:
        logger.info(
            f"🛡️ Quality Gate: {report.status.name} "
            f"(Reason: {report.reason}, "
            f"Confidence: {report.confidence_score:.2f})"
        )


# ============================================================
# TEST HARNESS
# ============================================================

if __name__ == "__main__":
    gate = RetrievalQualityGate()

    # --- Comparison: Missing one entity ---
    comparison_chunks = [
        {
            "content": "Far Cry 5 is an open-world shooter.",
            "source_title": "Far Cry 5 Review",
            "score": 0.82,
            "retrieval_context": "Far Cry 5",
        }
    ]

    # --- Listicle: Only headers ---
    listicle_chunks = [
        {
            "content": "Top things you should know about Far Cry 5",
            "source_title": "Top Far Cry 5 Guide",
            "chunk_index": 0,
            "score": 0.75,
        }
    ]

    # --- Noise: Sales articles ---
    noise_chunks = [
        {
            "content": "Huge discount and price drop in store sale",
            "source_title": "Far Cry 5 Sale Now Live",
            "score": 0.9,
        }
    ]

    print("\n=== QUALITY GATE TESTS ===\n")

    gate.evaluate(
        query="Compare Assassin's Creed Valhalla vs Far Cry 5",
        task=TaskType.COMPARISON,
        chunks=comparison_chunks,
    )

    gate.evaluate(
        query="Top 10 things to do in Far Cry 5",
        task=TaskType.LISTICLE,
        chunks=listicle_chunks,
    )

    gate.evaluate(
        query="Is Far Cry 5 on sale?",
        task=TaskType.OPEN,
        chunks=noise_chunks,
    )
