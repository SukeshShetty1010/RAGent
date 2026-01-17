# ============================================================
# tests/evaluation_runner.py
# Deterministic Behavioral + Metric Evaluation Harness (FINAL)
# ============================================================

from __future__ import annotations

import logging
import re
import time
import json
from dataclasses import dataclass
from typing import List, Pattern, Any

from agent.task_router import TaskRouter, TaskType
from retriever.orchestrator import RetrievalOrchestrator
from agent.prompt_manager import PromptManager

from tests.observability import ProfileBlock, MetricsRegistry
from tests.evaluation_metrics import (
    calculate_precision_at_k,
    calculate_compression_ratio,
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
    query: str
    expected_task: TaskType
    expected_web_trigger: bool
    required_structure_pattern: Pattern[str]
    expected_source_titles: List[str]


@dataclass
class EvaluationResult:
    query: str
    routing_accuracy: bool
    web_trigger_accuracy: bool
    structure_compliance: bool
    latency_ms: float
    precision_at_k: float
    compression_ratio: float


# ============================================================
# Evaluation Runner
# ============================================================

class EvaluationRunner:
    """
    Deterministic evaluation runner for RAG behavioral correctness
    and portfolio-grade metrics.
    """

    def __init__(self) -> None:
        self.router = TaskRouter()
        self.orchestrator = RetrievalOrchestrator()
        self.prompt_manager = PromptManager()

    # --------------------------------------------------------

    def run_suite(self, test_cases: List[TestCase]) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        try:
            for tc in test_cases:
                logger.info("\n" + "-" * 80)
                logger.info(f"Evaluating query: {tc.query}")

                registry = MetricsRegistry.get()

                with ProfileBlock("REQUEST_TOTAL"):
                    start = time.perf_counter()

                    # ---------------- Routing ----------------
                    decision = self.router.route(tc.query)
                    routing_ok = decision.task == tc.expected_task

                    # ---------------- Retrieval ----------------
                    chunks, merge_state, _quality = self.orchestrator.run(
                        query=tc.query,
                        task=decision.task,
                        config=self._config_from_decision(decision),
                    )

                    web_triggered = merge_state in {
                        "LOCAL_PLUS_WEB",
                        "LOCAL_WEB_ATTEMPTED",
                    }
                    web_ok = web_triggered == tc.expected_web_trigger

                    # ---------------- Prompt ----------------
                    prompt = self.prompt_manager.generate_prompt(
                        query=tc.query,
                        chunks=chunks,
                        task=decision.task,
                    )

                    structure_ok = bool(
                        tc.required_structure_pattern.search(prompt)
                    )

                    latency_ms = round(
                        (time.perf_counter() - start) * 1000.0, 2
                    )

                # =================================================
                # METRICS
                # =================================================

                precision = calculate_precision_at_k(
                    retrieved_chunks=chunks,
                    expected_source_titles=tc.expected_source_titles,
                    k=min(5, len(chunks)),
                )

                # Honest compression fallback:
                # until retrieval_results_count is instrumented,
                # we cannot know the true raw retrieval size.
                raw_retrieved = len(chunks)

                compression = calculate_compression_ratio(
                    initial_retrieved_count=raw_retrieved,
                    final_assembled_count=len(chunks),
                )

                results.append(
                    EvaluationResult(
                        query=tc.query,
                        routing_accuracy=routing_ok,
                        web_trigger_accuracy=web_ok,
                        structure_compliance=structure_ok,
                        latency_ms=latency_ms,
                        precision_at_k=precision.precision,
                        compression_ratio=compression.ratio,
                    )
                )

                logger.info(f"Routing OK: {routing_ok}")
                logger.info(f"Web Trigger OK: {web_ok}")
                logger.info(f"Structure OK: {structure_ok}")
                logger.info(f"Precision@K: {precision.precision}")
                logger.info(f"Compression Ratio: {compression.ratio}")
                logger.info(f"Latency: {latency_ms:.2f} ms")

                print(
                    "\n--- LATENCY PROFILE ---\n",
                    json.dumps(
                        analyze_latency_profile(
                            registry.generate_report()
                        ),
                        indent=2,
                    ),
                )

            return results

        finally:
            # Prevent socket leaks
            try:
                self.orchestrator.close()
            except Exception:
                pass

    # --------------------------------------------------------

    @staticmethod
    def _config_from_decision(decision: Any) -> Any:
        class _Config:
            limit = decision.max_results
            allow_web_fallback = decision.web_search_allowed
            use_query_decomposition = (
                decision.retrieval_strategy == "decomposition"
            )
            use_window_expansion = (
                decision.retrieval_strategy == "window_expansion"
            )

        return _Config()


# ============================================================
# Test Harness
# ============================================================

if __name__ == "__main__":
    runner = EvaluationRunner()

    TEST_SUITE: List[TestCase] = [
        TestCase(
            query="Compare Assassin's Creed Valhalla vs Far Cry 5",
            expected_task=TaskType.COMPARISON,
            expected_web_trigger=False,
            expected_source_titles=[
                "Assassin's Creed Valhalla",
                "Far Cry 5",
            ],
            required_structure_pattern=re.compile(
                r"\*\*Overview\*\*.*\*\*Gameplay\*\*", re.S
            ),
        ),
        TestCase(
            query="What is the release date of Far Cry 5?",
            expected_task=TaskType.FACTUAL,
            expected_web_trigger=False,
            expected_source_titles=["Far Cry 5"],
            required_structure_pattern=re.compile(
                r"Answer the user's question concisely", re.S
            ),
        ),
    ]

    results = runner.run_suite(TEST_SUITE)

    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps([r.__dict__ for r in results], indent=2))
