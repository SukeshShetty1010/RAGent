# ============================================================
# agents/task_router.py
# Deterministic Task Router (Brain Stem)
# ============================================================

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from enum import Enum
from typing import List


# ============================================================
# ENUMS
# ============================================================

class TaskType(Enum):
    """
    Strict task classification enum (priority-ordered externally).
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
    reason: str
    retrieval_strategy: str
    web_search_allowed: bool
    max_results: int


# ============================================================
# ROUTER
# ============================================================

class TaskRouter:
    """
    Deterministic, stateless task router.

    - O(1) regex checks
    - No LLM
    - No DB
    - Pure heuristic brain-stem
    """

    # --------------------------------------------------------
    # Regex triggers (compiled once → O(1))
    # --------------------------------------------------------

    _COMPARISON_PATTERNS: List[re.Pattern] = [
        re.compile(r"\bvs\b"),
        re.compile(r"\bversus\b"),
        re.compile(r"\bcompare\b"),
        re.compile(r"\bdifference between\b"),
        re.compile(r"\bbetter than\b"),
        re.compile(r"\bworse than\b"),
    ]

    _LISTICLE_PATTERNS: List[re.Pattern] = [
        re.compile(r"\btop\s*\d+\b"),
        re.compile(r"\bbest\b"),
        re.compile(r"\bworst\b"),
        re.compile(r"\blist\b"),
        re.compile(r"\branking\b"),
        re.compile(r"\brecommendations\b"),
        re.compile(r"\bthings to do\b"),
        re.compile(r"\btips\b"),
    ]

    _FACTUAL_PATTERNS: List[re.Pattern] = [
        re.compile(r"\bwhat is\b"),
        re.compile(r"\bwho is\b"),
        re.compile(r"\bwhen did\b"),
        re.compile(r"\brelease date\b"),
        re.compile(r"\bdeveloper\b"),
        re.compile(r"\bpublisher\b"),
        re.compile(r"\bspecs\b"),
        re.compile(r"\brequirements\b"),
    ]

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def route(self, query: str) -> RouterDecision:
        """
        Route a user query into a deterministic task type.
        Fail-safe: always returns a RouterDecision.
        """

        try:
            normalized = self._normalize(query)

            # ------------------------------------------------
            # Priority 1: COMPARISON
            # ------------------------------------------------
            for pattern in self._COMPARISON_PATTERNS:
                if pattern.search(normalized):
                    return RouterDecision(
                        task=TaskType.COMPARISON,
                        reason=f"Matched comparison trigger: '{pattern.pattern}'",
                        retrieval_strategy="decomposition",
                        web_search_allowed=False,
                        max_results=5,
                    )

            # ------------------------------------------------
            # Priority 2: LISTICLE
            # ------------------------------------------------
            for pattern in self._LISTICLE_PATTERNS:
                if pattern.search(normalized):
                    return RouterDecision(
                        task=TaskType.LISTICLE,
                        reason=f"Matched listicle trigger: '{pattern.pattern}'",
                        retrieval_strategy="window_expansion",
                        web_search_allowed=False,
                        max_results=10,
                    )

            # ------------------------------------------------
            # Priority 3: FACTUAL
            # ------------------------------------------------
            for pattern in self._FACTUAL_PATTERNS:
                if pattern.search(normalized):
                    return RouterDecision(
                        task=TaskType.FACTUAL,
                        reason=f"Matched factual trigger: '{pattern.pattern}'",
                        retrieval_strategy="standard",
                        web_search_allowed=False,
                        max_results=5,
                    )

            # ------------------------------------------------
            # Priority 4: OPEN (fallback)
            # ------------------------------------------------
            return RouterDecision(
                task=TaskType.OPEN,
                reason="No deterministic rule matched; fallback to OPEN",
                retrieval_strategy="hybrid",
                web_search_allowed=True,
                max_results=5,
            )

        except Exception as exc:
            # ------------------------------------------------
            # Absolute fail-safe
            # ------------------------------------------------
            return RouterDecision(
                task=TaskType.OPEN,
                reason=f"Router exception fallback: {exc}",
                retrieval_strategy="hybrid",
                web_search_allowed=True,
                max_results=5,
            )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    @staticmethod
    def _normalize(query: str) -> str:
        """
        Lowercase and strip punctuation.
        """
        if not isinstance(query, str):
            return ""

        text = query.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text


# ============================================================
# TEST HARNESS
# ============================================================

if __name__ == "__main__":
    router = TaskRouter()

    test_queries = [
        # Comparison beats listicle
        "Compare the top 10 RPG games of all time",
        # Pure listicle
        "Top 5 open world games to play in 2024",
        # Factual
        "What is the release date of Far Cry 5?",
        # Open-ended
        "Explain why Far Cry 5 is controversial",
    ]

    print("\n=== Task Router Test Output ===\n")
    for q in test_queries:
        decision = router.route(q)
        print(f"Query: {q}")
        print(decision)
        print("-" * 60)
