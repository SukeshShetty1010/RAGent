"""
agent/capability/capability_assessor.py

This module implements the **Honesty Gate** of the RAG system.

The CapabilityAssessor evaluates whether the system can *honestly*
fulfill a user request based on retrieved evidence and intent signals.

It is strictly evidence-driven and does NOT judge upstream routing
decisions or prompt strategies.
"""

import logging
from typing import Set, List, Dict, Any, DefaultDict
from collections import defaultdict

from agent.capability.capability_types import AnswerCapability
from agent.intent.intent_signals import IntentSignal
from retriever.quality_gate import QualityReport, QualityStatus
from utils.observability import MetricsRegistry

logger = logging.getLogger("RAG_CAPABILITY_ASSESSOR")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


class CapabilityAssessor:
    """
    Deterministic capability assessor.

    Grading model:
    - Start optimistic (FULL)
    - Downgrade to PARTIAL when coverage or integrity is incomplete
    - Downgrade to INSUFFICIENT when answering would be dishonest

    This assessor is:
    - Stateless
    - Deterministic
    - Evidence-driven
    - Fail-safe
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        *,
        intent_signals: Set[IntentSignal],
        evidence: List[Dict[str, Any]],
        quality: QualityReport,
    ) -> AnswerCapability:
        """
        Assess the system's ability to honestly answer a request.
        """

        try:
            quality_status = quality.status

            # ----------------------------------------------------------
            # Absolute honesty checks
            # ----------------------------------------------------------
            if not evidence or quality_status == QualityStatus.QUALITY_EMPTY:
                reason = "no_evidence" if not evidence else f"quality_empty:{quality.reason}"
                MetricsRegistry.get().record("capability_reason", reason)
                return AnswerCapability.INSUFFICIENT

            capability = AnswerCapability.FULL

            if quality_status == QualityStatus.QUALITY_WEAK:
                capability = AnswerCapability.PARTIAL

            # ----------------------------------------------------------
            # COMPARISON
            # ----------------------------------------------------------
            if IntentSignal.COMPARISON in intent_signals:
                entity_coverage = self._entity_coverage(evidence)

                if entity_coverage < 2:
                    MetricsRegistry.get().record(
                        "capability_reason", "comparison_entity_coverage"
                    )
                    return AnswerCapability.INSUFFICIENT

                if self._is_unbalanced(evidence):
                    capability = AnswerCapability.PARTIAL

            # ----------------------------------------------------------
            # LISTICLE — no dedicated rule.
            # A listicle with evidence is never INSUFFICIENT (guaranteed by
            # the check above), but it must not upgrade past a WEAK-evidence
            # downgrade either, so it simply keeps the `capability` value
            # quality has already determined.
            # ----------------------------------------------------------

            # ----------------------------------------------------------
            # TEMPORAL
            # ----------------------------------------------------------
            if IntentSignal.TEMPORAL in intent_signals:
                if not quality.has_temporal_signal:
                    capability = AnswerCapability.PARTIAL

            return capability

        except Exception:
            logger.exception("CapabilityAssessor.assess() failed unexpectedly")
            MetricsRegistry.get().inc("capability_assess_errors")
            # Fail-safe: degrade, never hallucinate
            return AnswerCapability.PARTIAL

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_coverage(evidence: List[Dict[str, Any]]) -> int:
        """
        Count distinct entities represented in evidence.
        """
        entities = set()

        for chunk in evidence:
            entity = (
                chunk.get("retrieval_context")
                or chunk.get("source_title")
            )
            if entity:
                entities.add(entity)

        return len(entities)

    @staticmethod
    def _is_unbalanced(
        evidence: List[Dict[str, Any]],
    ) -> bool:
        """
        Detect severe evidence imbalance across entities.

        Example:
            5 chunks for entity A, 1 chunk for entity B → unbalanced
        """
        counts: DefaultDict[str, int] = defaultdict(int)

        for chunk in evidence:
            entity = (
                chunk.get("retrieval_context")
                or chunk.get("source_title")
            )
            if entity:
                counts[entity] += 1

        if len(counts) < 2:
            return True

        max_chunks = max(counts.values())
        min_chunks = min(counts.values())

        # Heuristic: >3× imbalance is considered partial
        return max_chunks > (min_chunks * 3)
