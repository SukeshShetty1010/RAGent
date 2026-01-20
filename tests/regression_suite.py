# ============================================================
# tests/regression_suite.py
# Permanent Regression Memory for RAG Behavioral Guarantees
# (BUDGET-AWARE STRUCTURE CHECKS)
# ============================================================

from __future__ import annotations

import sys
import re
from dataclasses import dataclass
from typing import List

from agent.task_router import TaskType
from tests.evaluation_runner import (
    EvaluationRunner,
    TestCase,
    EvaluationResult,
)

# ------------------------------------------------------------
# ANSI colors
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

    This is not a unit test.
    This is a behavioral guarantee enforced forever.
    """

    bug_id: str
    description: str
    test_case: TestCase


# ============================================================
# Regression Vault (Permanent Memory)
# ============================================================

REGRESSION_VAULT: List[RegressionCase] = [
    RegressionCase(
        bug_id="BUG-001",
        description=(
            "Semantic intersection trap: comparison queries previously mixed "
            "evidence across entities, causing context soup and wrong task routing."
        ),
        test_case=TestCase(
            query="Compare Assassin's Creed Valhalla vs Far Cry 5",
            expected_task=TaskType.COMPARISON,
            expected_web_trigger=False,
            expected_source_titles=[
                "Assassin's Creed Valhalla",
                "Far Cry 5",
            ],
            # Accept verbose OR concise comparison structure
            required_structure_pattern=re.compile(
                r"(\*\*Overview\*\*.*\*\*Gameplay\*\*)|"
                r"(Gameplay, Story, World Design, Tone, Systems)",
                re.S,
            ),
        ),
    ),
    RegressionCase(
        bug_id="BUG-002",
        description=(
            "Listicle ordering regression: editorial chunks lost ordering and "
            "web/news articles polluted the ordered list."
        ),
        test_case=TestCase(
            query="Top 10 things to do in Far Cry 5",
            expected_task=TaskType.LISTICLE,
            expected_web_trigger=False,
            expected_source_titles=["Far Cry 5"],
            # Accept numbered list OR listicle phrasing
            required_structure_pattern=re.compile(
                r"(1\.\s+)|"
                r"(ordered list)|(Top\s+\d+)",
                re.I | re.S,
            ),
        ),
    ),
    RegressionCase(
        bug_id="BUG-003",
        description=(
            "Temporal knowledge failure: factual queries with time sensitivity "
            "did not trigger web augmentation when local data was stale."
        ),
        test_case=TestCase(
            query="Latest patch notes for Assassin's Creed Valhalla",
            expected_task=TaskType.OPEN,
            expected_web_trigger=True,
            expected_source_titles=["Assassin's Creed Valhalla"],
            # Accept any factual answer body
            required_structure_pattern=re.compile(
                r".{40,}", re.S
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
                f"Expected task {expected.expected_task.name}, routing failed"
            )

        if not actual.web_trigger_accuracy:
            diffs.append(
                "Web trigger behavior incorrect "
                f"(expected={expected.expected_web_trigger})"
            )

        if not actual.structure_compliance:
            diffs.append("Prompt structure constraint violated")

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
