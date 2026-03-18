# ============================================================
# agent/task_router.py
# Intent-Aware Deterministic Task Router (CACHE-SAFE)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Set

from tests.observability import ProfileBlock, MetricsRegistry
from tests.caching import cacheable

from agent.intent.intent_signals import IntentSignal
from agent.intent.intent_extractor import IntentSignalExtractor


# ============================================================
# INTENT SCHEMA VERSIONING
# ============================================================

INTENT_SCHEMA_VERSION = "v2"
# ⬆️ Bump this whenever intent extraction logic changes


# ============================================================
# ENUMS
# ============================================================

class TaskType(Enum):
    """
    Primary execution modes for retrieval and prompting.
    """
    COMPARISON = "comparison"
    LISTICLE = "listicle"
    FACTUAL = "factual"
    OPEN = "open"


# ============================================================
# CONTRACT
# ============================================================

@dataclass(frozen=True)
class RouterDecision:
    """
    Immutable routing decision contract.
    """
    task: TaskType
    intent_signals: Set[IntentSignal]
    reason: str
    retrieval_strategy: str
    web_search_allowed: bool
    max_results: int


# ============================================================
# ROUTER
# ============================================================

class TaskRouter:
    """
    Intent-aware, deterministic, cache-safe task router.

    ✔ Cache-safe with schema versioning
    ✔ Deterministic
    ✔ Observable
    ✔ No feasibility or quality reasoning
    """

    def __init__(self) -> None:
        self._intent_extractor = IntentSignalExtractor()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    @cacheable(ttl_seconds=3600)
    def route(
        self,
        query: str,
        intent_schema_version: str = INTENT_SCHEMA_VERSION,
    ) -> RouterDecision:
        """
        Route a user query into a primary TaskType.

        IMPORTANT:
        intent_schema_version is intentionally part of the
        cache key (handled correctly by updated cacheable).
        """

        with ProfileBlock("TaskRouting"):
            intent_signals = self._intent_extractor.extract(query)

            task = self._select_primary_task(intent_signals)

            decision = RouterDecision(
                task=task,
                intent_signals=intent_signals,
                reason=self._explain(task, intent_signals),
                retrieval_strategy=self._retrieval_strategy(task),
                web_search_allowed=self._web_allowed(task),
                max_results=self._max_results(task),
            )

            MetricsRegistry.get().record(
                "routing_decisions",
                decision.task.value,
            )

            return decision  

    # --------------------------------------------------------
    # Routing Policy
    # --------------------------------------------------------

    @staticmethod
    def _select_primary_task(
        intent_signals: Set[IntentSignal],
    ) -> TaskType:
        """
        Select the most expressive task given intent signals.
        Priority order matters.
        """

        if IntentSignal.COMPARISON in intent_signals:
            return TaskType.COMPARISON

        if IntentSignal.LISTICLE in intent_signals:
            return TaskType.LISTICLE

        if IntentSignal.FACTUAL in intent_signals:
            return TaskType.FACTUAL

        return TaskType.OPEN

    # --------------------------------------------------------
    # Retrieval Policy
    # --------------------------------------------------------

    @staticmethod
    def _retrieval_strategy(task: TaskType) -> str:
        if task == TaskType.COMPARISON:
            return "decomposition"
        if task == TaskType.LISTICLE:
            return "window_expansion"
        if task == TaskType.FACTUAL:
            return "standard"
        return "hybrid"

    @staticmethod
    def _web_allowed(task: TaskType) -> bool:
        return task == TaskType.OPEN

    @staticmethod
    def _max_results(task: TaskType) -> int:
        return 10 if task == TaskType.LISTICLE else 5

    # --------------------------------------------------------
    # Explainability
    # --------------------------------------------------------

    @staticmethod
    def _explain(
        task: TaskType,
        intent_signals: Set[IntentSignal],
    ) -> str:
        if not intent_signals:
            return "No intent signals detected; fallback to OPEN"

        signals = ", ".join(sorted(s.value for s in intent_signals))
        return (
            f"Selected {task.value} as primary task "
            f"based on intent signals: {signals}"
        )


# ============================================================
# LOCAL TEST HARNESS
# ============================================================

if __name__ == "__main__":
    router = TaskRouter()

    queries = [
        "what is the comparison in the storyline between Far Cry 5 and Assassins Creed Valhalla?",
        "Top 5 open world games to play in 2024",
        "What is the release date of Far Cry 5?",
        "Explain why Far Cry 5 is controversial",
    ]

    for q in queries:
        print(router.route(q))
