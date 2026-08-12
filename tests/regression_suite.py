# ============================================================
# tests/regression_suite.py
# Permanent Regression Memory for RAG Behavioral Guarantees
# (CAPABILITY-AWARE)
# ============================================================

from __future__ import annotations

import sys
import re
from dataclasses import dataclass
from typing import List

from agent.task_router import TaskType
from agent.capability.capability_types import AnswerCapability
from tests.evaluation_runner import (
    EvaluationRunner,
    TestCase,
    EvaluationResult,
)

# ------------------------------------------------------------
# ANSI colors (CLI feedback)
# ------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# ============================================================
# Regression Case Contract
# ============================================================

@dataclass(frozen=True)
class RegressionCase:
    """
    Represents a historical bug that MUST never reappear.

    These are behavioral guarantees, not unit tests.
    """
    bug_id: str
    description: str
    test_case: TestCase


# ============================================================
# Regression Vault (Permanent Memory)
# ============================================================

REGRESSION_VAULT: List[RegressionCase] = [

    # --------------------------------------------------------
    # BUG-001: Mixed-intent comparison collapsed to factual
    # --------------------------------------------------------
    RegressionCase(
        bug_id="BUG-001",
        description=(
            "Mixed-intent comparison queries were incorrectly routed "
            "as FACTUAL, producing misleading answers instead of honest "
            "partial comparisons."
        ),
        test_case=TestCase(
            query="What is the comparison and differences between Far Cry 5 and Assassin’s Creed Valhalla",
            expected_task=TaskType.COMPARISON,
            # Phase 3.5 fixed quality_gate.py's is_noise substring-containment
            # bug (e.g. "wholesale" tripping the "sale" keyword) and replaced
            # the RRF-score threshold with a calibrated cross-encoder floor.
            # Both retain more of this query's genuine evidence than before,
            # producing balanced entity coverage -> FULL is the honest grade
            # now, not a weakened one. See flagship.md Phase 3.5.
            expected_capability=AnswerCapability.FULL,
            expected_source_titles=[
                "Far Cry 5",
                "Assassin’s Creed Valhalla",
            ],
            required_structure_pattern=re.compile(
                r"(Gameplay|Story|World Design|Tone|Systems)",
                re.I,
            ),
        ),
    ),

    # --------------------------------------------------------
    # BUG-002: Listicle ordering corruption
    # --------------------------------------------------------
    RegressionCase(
        bug_id="BUG-002",
        description=(
            "Listicle queries lost ordering guarantees or polluted "
            "results with unrelated sources."
        ),
        test_case=TestCase(
            query="Top 5 things to do in Far Cry 5",
            expected_task=TaskType.LISTICLE,
            expected_capability=AnswerCapability.FULL,
            expected_source_titles=["Far Cry 5"],
            required_structure_pattern=re.compile(
                r"(1\.\s+)|(Top\s+\d+)",
                re.I,
            ),
        ),
    ),

    # --------------------------------------------------------
    # BUG-003: Temporal factual answers hallucinated freshness
    # --------------------------------------------------------
    RegressionCase(
        bug_id="BUG-003",
        description=(
            "Temporal or update-seeking queries previously hallucinated "
            "freshness instead of degrading honestly."
        ),
        test_case=TestCase(
            query="Latest update for Assassin’s Creed Valhalla",
            expected_task=TaskType.OPEN,
            # Phase 3.5: quality_gate.py now reports QUALITY_OK with a real
            # temporal signal for this query (calibrated relevance floor +
            # is_noise word-boundary fix), and capability_assessor.py's
            # TEMPORAL rule only downgrades to PARTIAL when has_temporal_signal
            # is False. Evidence is genuinely sufficient here -> FULL is
            # correct, not a regression. See flagship.md Phase 3.5.
            expected_capability=AnswerCapability.FULL,
            expected_source_titles=["Assassin’s Creed Valhalla"],
            required_structure_pattern=re.compile(
                r"(latest|update|patch|version)",
                re.I,
            ),
        ),
    ),
]


# ============================================================
# Regression Runner
# ============================================================

class RegressionRunner:
    """
    Executes the regression vault against the current system.

    Philosophy:
    - Fail fast
    - Fail loud
    - Never forget fixed bugs
    """

    def __init__(self) -> None:
        self.evaluator = EvaluationRunner()

    # --------------------------------------------------------

    def run(self) -> bool:
        print("\n=== REGRESSION SUITE: PERMANENT MEMORY ===\n")

        all_passed = True

        for case in REGRESSION_VAULT:
            print(f"{YELLOW}{case.bug_id}{RESET}: {case.description}")

            result = self._run_case(case)

            if result is None:
                print(
                    f"{RED}❌ FAILED: No evaluation result returned{RESET}\n"
                )
                all_passed = False
                continue

            passed, diff = self._compare(case.test_case, result)

            if passed:
                print(f"{GREEN}✅ PASS{RESET}\n")
            else:
                print(f"{RED}❌ FAIL{RESET}")
                for line in diff:
                    print(f"   {RED}- {line}{RESET}")
                print()
                all_passed = False

        return all_passed

    # --------------------------------------------------------

    def _run_case(
        self,
        case: RegressionCase,
    ) -> EvaluationResult | None:
        results = self.evaluator.run_suite([case.test_case])
        return results[0] if results else None

    def _compare(
        self,
        expected: TestCase,
        actual: EvaluationResult,
    ) -> tuple[bool, List[str]]:

        diffs: List[str] = []

        if not actual.routing_accuracy:
            diffs.append(
                f"Routing failed (expected {expected.expected_task.name})"
            )

        if not actual.capability_accuracy:
            diffs.append(
                f"Capability mismatch (expected {expected.expected_capability.value})"
            )

        if not actual.structure_compliance:
            diffs.append("Answer structure constraint violated")

        return (len(diffs) == 0, diffs)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    runner = RegressionRunner()
    success = runner.run()

    if success:
        print(f"{GREEN}ALL REGRESSIONS PASSED — SYSTEM STABLE{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}REGRESSION FAILURE — FIX REQUIRED{RESET}")
        sys.exit(1)
