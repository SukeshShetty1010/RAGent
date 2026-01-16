# ============================================================
# retriever/strategy_selector.py
# Step 2: Retrieval Strategy Selection
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

from agent.task_router import RouterDecision, TaskType


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

    This is a pure configuration factory.
    """

    def select(self, decision: RouterDecision) -> RetrievalConfiguration:
        """
        Translate a RouterDecision into a RetrievalConfiguration.
        """

        if decision.task == TaskType.COMPARISON:
            # Compare independent entities
            return RetrievalConfiguration(
                limit=5,
                use_window_expansion=False,
                use_query_decomposition=True,
                allow_web_fallback=False,
            )

        if decision.task == TaskType.LISTICLE:
            # Ordered, continuous content
            return RetrievalConfiguration(
                limit=10,
                use_window_expansion=True,
                use_query_decomposition=False,
                allow_web_fallback=False,
            )

        if decision.task == TaskType.FACTUAL:
            # High-precision single answer
            return RetrievalConfiguration(
                limit=5,
                use_window_expansion=False,
                use_query_decomposition=False,
                allow_web_fallback=False,
            )

        # Fallback: OPEN
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
    selector = StrategySelector()

    test_decisions = [
        RouterDecision(
            task=TaskType.COMPARISON,
            reason="Test comparison",
            retrieval_strategy="decomposition",
            web_search_allowed=False,
            max_results=5,
        ),
        RouterDecision(
            task=TaskType.LISTICLE,
            reason="Test listicle",
            retrieval_strategy="window_expansion",
            web_search_allowed=False,
            max_results=10,
        ),
        RouterDecision(
            task=TaskType.task,
            reason="Test factual",
            retrieval_strategy="standard",
            web_search_allowed=False,
            max_results=5,
        ),
        RouterDecision(
            task=TaskType.OPEN,
            reason="Test open",
            retrieval_strategy="hybrid",
            web_search_allowed=True,
            max_results=5,
        ),
    ]

    print("\n=== Strategy Selector Test Output ===\n")

    for decision in test_decisions:
        config = selector.select(decision)
        print(f"TaskType: {decision.task.name}")
        print(config)
        print("-" * 60)
