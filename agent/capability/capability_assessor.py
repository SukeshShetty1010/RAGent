"""
agent/capability/capability_assessor.py

This module implements the **Honesty Gate** of the RAG system.

The CapabilityAssessor evaluates whether the system can *honestly*
fulfill a user request based on retrieved evidence and intent signals.

It is strictly evidence-driven and does NOT judge upstream routing
decisions or prompt strategies.
"""

from typing import Set, List, Dict, Any, DefaultDict
from collections import defaultdict

from agent.capability.capability_types import AnswerCapability
from agent.intent.intent_signals import IntentSignal
from retriever.quality_gate import QualityStatus


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
        quality: Dict[str, Any],
    ) -> AnswerCapability:
        """
        Assess the system's ability to honestly answer a request.

        Args:
            intent_signals: Detected IntentSignal set
            evidence: Retrieved evidence chunks
            quality: Quality diagnostics containing:
                - status: QualityStatus
                - has_temporal_signal: bool

        Returns:
            AnswerCapability enum value.
        """

        # --------------------------------------------------------------
        # Absolute honesty checks
        # --------------------------------------------------------------
        try:
            quality_status = quality.get("status")

            if not evidence or quality_status == QualityStatus.QUALITY_EMPTY:
                return AnswerCapability.INSUFFICIENT

            capability = AnswerCapability.FULL

            if quality_status == QualityStatus.QUALITY_WEAK:
                capability = AnswerCapability.PARTIAL

            # ----------------------------------------------------------
            # Intent-specific feasibility checks (evidence-based only)
            # ----------------------------------------------------------

            # ------------------------------
            # COMPARISON
            # ------------------------------
            if IntentSignal.COMPARISON in intent_signals:
                entity_coverage = self._entity_coverage(evidence)

                if entity_coverage < 2:
                    # Cannot compare fewer than 2 entities honestly
                    return AnswerCapability.INSUFFICIENT

                if self._is_unbalanced(entity_coverage, evidence):
                    capability = AnswerCapability.PARTIAL

            # ------------------------------
            # LISTICLE
            # ------------------------------
            if IntentSignal.LISTICLE in intent_signals:
                if len(evidence) < 3:
                    capability = AnswerCapability.PARTIAL

            # ------------------------------
            # TEMPORAL
            # ------------------------------
            if IntentSignal.TEMPORAL in intent_signals:
                if not quality.get("has_temporal_signal", False):
                    capability = AnswerCapability.PARTIAL

            return capability

        except Exception:
            # ----------------------------------------------------------
            # Fail-safe: degrade, never hallucinate
            # ----------------------------------------------------------
            return AnswerCapability.PARTIAL

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_coverage(evidence: List[Dict[str, Any]]) -> int:
        """
        Count distinct entities represented in evidence.

        Uses `retrieval_context` if available (comparison decomposition),
        otherwise falls back to source_title grouping.
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
        entity_count: int,
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
