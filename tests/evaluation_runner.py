# ============================================================
# tests/evaluation_runner.py
# Capability-Aware End-to-End Evaluation Runner (FINAL)
# ============================================================

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import List, Pattern

from engine.execution_engine import RageEngine
from agent.task_router import TaskType
from agent.capability.capability_types import AnswerCapability

from utils.observability import ProfileBlock
from tests.evaluation_metrics import (
    calculate_precision_at_k,
    analyze_latency_profile,
)

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("RAG_EVALUATION_RUNNER")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Data Contracts
# ============================================================

@dataclass(frozen=True)
class TestCase:
    """
    Declarative evaluation contract.
    """
    query: str
    expected_task: TaskType
    expected_capability: AnswerCapability
    required_structure_pattern: Pattern[str]
    expected_source_titles: List[str]


@dataclass(frozen=True)
class EvaluationResult:
    """
    Atomic evaluation outcome for a single query.
    """
    query: str
    routing_accuracy: bool
    capability_accuracy: bool
    structure_compliance: bool
    latency_ms: float
    precision_at_k: float
    answer_capability: str


# ============================================================
# Evaluation Runner
# ============================================================

class EvaluationRunner:
    """
    Capability-aware, end-to-end evaluation harness.

    Evaluates:
    - Intent → Task routing correctness
    - Capability honesty (FULL / PARTIAL / INSUFFICIENT)
    - Prompt structural compliance
    - Retrieval precision
    - Latency characteristics

    IMPORTANT:
    This runner exercises the *exact same execution path*
    as production.
    """

    def __init__(self) -> None:
        self.engine = RageEngine()

    # --------------------------------------------------------

    def run_suite(
        self,
        test_cases: List[TestCase],
    ) -> List[EvaluationResult]:

        results: List[EvaluationResult] = []

        for tc in test_cases:
            logger.info("\n" + "=" * 80)
            logger.info(f"Evaluating query: {tc.query}")

            start = time.perf_counter()

            with ProfileBlock("REQUEST_TOTAL"):
                execution = self.engine.run(tc.query)

            latency_ms = round(
                (time.perf_counter() - start) * 1000.0,
                2,
            )

            agent = execution["agent_decisions"]
            kpis = execution["kpis"]
            evidence = execution["evidence"]
            answer = execution["final_answer"]

            # ---------------- Routing ----------------
            routing_ok = (
                agent.get("task")
                == tc.expected_task.value
            )

            # ---------------- Capability ----------------
            actual_capability = agent.get("answer_capability")
            capability_ok = (
                actual_capability
                == tc.expected_capability.value
            )

            # ---------------- Structure ----------------
            structure_ok = bool(
                tc.required_structure_pattern.search(answer)
            )

            # ---------------- Precision ----------------
            precision = calculate_precision_at_k(
                retrieved_chunks=evidence,
                expected_source_titles=tc.expected_source_titles,
                k=min(5, len(evidence)),
            )

            results.append(
                EvaluationResult(
                    query=tc.query,
                    routing_accuracy=routing_ok,
                    capability_accuracy=capability_ok,
                    structure_compliance=structure_ok,
                    latency_ms=latency_ms,
                    precision_at_k=precision.precision,
                    answer_capability=actual_capability,
                )
            )

            # ---------------- Logging ----------------
            logger.info(f"Routing OK: {routing_ok}")
            logger.info(f"Capability OK: {capability_ok}")
            logger.info(f"Structure OK: {structure_ok}")
            logger.info(f"Precision@K: {precision.precision}")
            logger.info(f"Latency: {latency_ms:.2f} ms")
            logger.info(f"Answer Capability: {actual_capability}")
            logger.info(f"LLM Ran: {kpis.get('llm_ran')}")

            print(
                "\n--- LATENCY PROFILE ---\n",
                json.dumps(
                    analyze_latency_profile(
                        execution["raw_metrics"]
                    ),
                    indent=2,
                ),
            )

        return results

    # --------------------------------------------------------

    def close(self) -> None:
        self.engine.close()


# ============================================================
# Standalone Test Harness
# ============================================================

if __name__ == "__main__":
    import re

    runner = EvaluationRunner()

    TEST_SUITE: List[TestCase] = [
        TestCase(
            query=(
                "What is the comparison and differences between "
                "Far Cry 5 and Assassin’s Creed Valhalla"
            ),
            expected_task=TaskType.COMPARISON,
            expected_capability=AnswerCapability.PARTIAL,
            expected_source_titles=[
                "Far Cry 5",
                "Assassin’s Creed Valhalla",
            ],
            required_structure_pattern=re.compile(
                r"(Gameplay|Story|World Design|Tone|Systems)",
                re.I,
            ),
        ),
        TestCase(
            query="What is the release date of Far Cry 5?",
            expected_task=TaskType.FACTUAL,
            expected_capability=AnswerCapability.FULL,
            expected_source_titles=["Far Cry 5"],
            required_structure_pattern=re.compile(
                r"\d{4}",
                re.I,
            ),
        ),
    ]

    results = runner.run_suite(TEST_SUITE)

    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps([r.__dict__ for r in results], indent=2))

    runner.close()
