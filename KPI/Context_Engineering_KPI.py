# ============================================================
# tests/KPI/Context_Engineering_KPI.py
# RESUME-GRADE KPI: CONTEXT ENGINEERING
# ============================================================

from __future__ import annotations

import logging
from typing import List, Tuple

from engine.execution_engine import RageEngine
from tests.observability import MetricsRegistry
from tests.evaluation_metrics import calculate_compression_ratio
from agent.task_router import TaskType

# ------------------------------------------------------------
# ANSI formatting (resume-grade CLI table)
# ------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# ============================================================
# Standard Traffic (from KPI_run.py)
# ============================================================

TRAFFIC: List[Tuple[str, TaskType]] = [
    ("Compare Far Cry 5 vs Assassin’s Creed Valhalla", TaskType.COMPARISON),
    ("Top 5 things to do in Far Cry 5", TaskType.LISTICLE),
    ("What is the release date of Far Cry 5?", TaskType.FACTUAL),
    ("Explain why Far Cry 5 is controversial", TaskType.OPEN),
    ("Latest update for Assassin’s Creed Valhalla", TaskType.OPEN),
]


# ============================================================
# KPI Runner
# ============================================================

class ContextEngineeringKPI:
    """
    Targeted KPI runner for Context Engineering.

    Metrics:
    - Context Compression Ratio (Evidence Compression)
    - Redundancy Rejection Rate (Jaccard-based)
    - Prompt Budget Compliance Rate

    Purpose:
    - Prove context discipline
    - Demonstrate redundancy filtering
    - Demonstrate guardrail enforcement for resumes / exec review
    """

    def __init__(self) -> None:
        self.engine = RageEngine()
        self.registry = MetricsRegistry.get()

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def run(self) -> None:
        logging.getLogger().setLevel(logging.WARNING)

        # Reset registry before run (important for clean aggregation)
        self.registry._counters.clear()
        self.registry._distributions.clear()
        self.registry._categoricals.clear()

        # =====================================================
        # Traffic Execution
        # =====================================================

        for query, _ in TRAFFIC:
            self.engine.run(query)

        # =====================================================
        # Data Extraction (Registry)
        # =====================================================

        # ---- Context Compression ----
        input_chunks = sum(
            self.registry._distributions
            .get("context_input_chunks", {})
            .values
        )
        final_chunks = sum(
            self.registry._distributions
            .get("context_final_chunks", {})
            .values
        )

        compression = calculate_compression_ratio(
            initial_retrieved_count=input_chunks,
            final_assembled_count=final_chunks,
        )

        # ---- Redundancy Rejection Rate ----
        # Percentage of deduplicated candidate chunks
        # rejected by Jaccard-based redundancy filtering

        redundant_rejections_metric = self.registry._counters.get(
            "context_redundant_rejections"
        )

        redundant_rejections = (
            redundant_rejections_metric.value
            if redundant_rejections_metric else 0
        )

        redundancy_candidates = sum(
            self.registry._distributions
            .get("context_deduped_chunks", {})
            .values
        )

        redundancy_rate = (
            round(
                (redundant_rejections / redundancy_candidates) * 100,
                2,
            )
            if redundancy_candidates else 0.0
        )

        # =====================================================
        # Prompt Budget Compliance
        # =====================================================

        prompt_budget_metric = self.registry._categoricals.get(
            "prompt_budget_mode"
        )
        prompt_mode_metric = self.registry._categoricals.get(
            "prompt_mode"
        )

        budget_mode_count = (
            sum(prompt_budget_metric.values.values())
            if prompt_budget_metric else 0
        )

        insufficient_count = (
            prompt_mode_metric.values.get("insufficient", 0)
            if prompt_mode_metric else 0
        )

        compliant_prompts = budget_mode_count + insufficient_count
        total_runs = len(TRAFFIC)

        prompt_budget_compliance_rate = (
            round((compliant_prompts / total_runs) * 100, 2)
            if total_runs else 0.0
        )

        # =====================================================
        # Dashboard Output
        # =====================================================

        print(f"\n{BOLD}{CYAN}RESUME-GRADE KPI: CONTEXT ENGINEERING{RESET}\n")

        header = (
            f"{BOLD}| Metric Name                     "
            f"| Value    | Target   |{RESET}"
        )
        divider = "-" * len(header)

        print(divider)
        print(header)
        print(divider)

        print(
            f"| Context Compression Ratio       "
            f"| {compression.ratio:<6.2f}x "
            f"| < 0.4x   |"
        )
        print(
            f"| Redundancy Rejection Rate       "
            f"| {redundancy_rate:>6.2f}% "
            f"| > 10%    |"
        )
        print(
            f"| Prompt Budget Compliance Rate   "
            f"| {prompt_budget_compliance_rate:>6.2f}% "
            f"| = 100%   |"
        )

        print(divider)
        print()


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    ContextEngineeringKPI().run()
