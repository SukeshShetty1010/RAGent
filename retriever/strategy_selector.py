# ============================================================
# retriever/strategy_selector.py
# Intent-Aware Retrieval Strategy Selector
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from agent.task_router import RouterDecision, TaskType
from agent.intent.intent_signals import IntentSignal


# ============================================================
# DATA CONTRACT
# ============================================================

@dataclass(frozen=True)
class RetrievalConfiguration:
    """
    Concrete retrieval configuration derived from a RouterDecision.

    This object contains ONLY execution settings.
    It performs no retrieval and makes no external calls.
    """
    limit: int
    use_window_expansion: bool
    use_query_decomposition: bool
    allow_web_fallback: bool


# ============================================================
# STRATEGY SELECTOR
# ============================================================

class StrategySelector:
    """
    Deterministic mapping from RouterDecision → RetrievalConfiguration.

    This selector is:
    - Intent-aware
    - Optimistic (prefers wider retrieval when in doubt)
    - Pure (no side effects, no I/O)

    It answers one question:
        "How should we retrieve evidence for this request?"
    """

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def select(self, decision: RouterDecision) -> RetrievalConfiguration:
        """
        Translate a RouterDecision into a RetrievalConfiguration.
        """

        task = decision.task
        intent_signals: Set[IntentSignal] = decision.intent_signals
        has_temporal = IntentSignal.TEMPORAL in intent_signals

        # ----------------------------------------------------
        # COMPARISON (highest evidence demand)
        # ----------------------------------------------------
        if task == TaskType.COMPARISON:
            return RetrievalConfiguration(
                limit=5,
                use_window_expansion=False,
                use_query_decomposition=True,
                allow_web_fallback=has_temporal,
            )

        # ----------------------------------------------------
        # LISTICLE
        # ----------------------------------------------------
        if task == TaskType.LISTICLE:
            return RetrievalConfiguration(
                limit=10,
                use_window_expansion=True,
                use_query_decomposition=False,
                allow_web_fallback=has_temporal,
            )

        # ----------------------------------------------------
        # FACTUAL (may widen if mixed intent)
        # ----------------------------------------------------
        if task == TaskType.FACTUAL:
            # If other intents exist, prefer safer (wider) retrieval
            if len(intent_signals) > 1:
                return RetrievalConfiguration(
                    limit=7,
                    use_window_expansion=True,
                    use_query_decomposition=False,
                    allow_web_fallback=has_temporal,
                )

            return RetrievalConfiguration(
                limit=5,
                use_window_expansion=False,
                use_query_decomposition=False,
                allow_web_fallback=has_temporal,
            )

        # ----------------------------------------------------
        # OPEN (fallback / exploratory)
        # ----------------------------------------------------
        return RetrievalConfiguration(
            limit=5,
            use_window_expansion=False,
            use_query_decomposition=False,
            allow_web_fallback=True,
        )


# ============================================================
# TEST HARNESS
# ============================================================

if __name__ == "__main__":
    from agent.task_router import TaskType
    from agent.intent.intent_signals import IntentSignal

    selector = StrategySelector()

    test_decisions = [
        RouterDecision(
            task=TaskType.COMPARISON,
            intent_signals={IntentSignal.COMPARISON},
            reason="Test comparison",
        ),
        RouterDecision(
            task=TaskType.FACTUAL,
            intent_signals={IntentSignal.FACTUAL, IntentSignal.COMPARISON},
            reason="Mixed intent factual",
        ),
        RouterDecision(
            task=TaskType.OPEN,
            intent_signals=set(),
            reason="Fallback",
        ),
    ]

    print("\n=== Strategy Selector Test Output ===\n")
    for decision in test_decisions:
        config = selector.select(decision)
        print(f"TaskType: {decision.task.name}")
        print(f"IntentSignals: {[s.value for s in decision.intent_signals]}")
        print(config)
        print("-" * 60)
