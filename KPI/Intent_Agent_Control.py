# ============================================================
# tests/resume_kpi_dashboard.py
# Resume-Grade KPI Dashboard — Routing, Intent & Stability
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from engine.execution_engine import RageEngine
from agent.task_router import TaskType

# ------------------------------------------------------------
# ANSI formatting (clean, resume-grade)
# ------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"


# ============================================================
# Generic KPI Contract (Open for Extension)
# ============================================================

@dataclass
class KPIMetric:
    category: str
    name: str
    value: float     # percentage value (0–100)
    samples: int     # N (Samples)


# ============================================================
# Dashboard Runner
# ============================================================

class ResumeKPIDashboard:
    """
    Executive-facing KPI dashboard proving:
    - Intent correctness
    - Routing correctness
    - Routing determinism (stability across runs)
    """

    def __init__(self) -> None:
        self.engine = RageEngine()

        # Canonical traffic (stable evaluation set)
        self.TRAFFIC: List[Tuple[str, TaskType]] = [
            ("Compare Far Cry 5 vs Assassin’s Creed Valhalla", TaskType.COMPARISON),
            ("Top 5 things to do in Far Cry 5", TaskType.LISTICLE),
            ("What is the release date of Far Cry 5?", TaskType.FACTUAL),
            ("Explain why Far Cry 5 is controversial", TaskType.OPEN),
            ("Latest update for Assassin’s Creed Valhalla", TaskType.OPEN),
        ]

    # --------------------------------------------------------
    # Core KPI Computation (Double-Pass)
    # --------------------------------------------------------

    def run(self) -> List[KPIMetric]:
        """
        Execute traffic twice to derive accuracy + determinism KPIs.

        Returns:
            List[KPIMetric]
        """
        logging.getLogger().setLevel(logging.WARNING)

        total = len(self.TRAFFIC)

        # -------------------------------
        # PASS A — baseline routing
        # -------------------------------
        pass_a_results: Dict[str, str] = {}
        correct_routes = 0

        for query, expected_task in self.TRAFFIC:
            result = self.engine.run(query)
            actual_task = result["agent_decisions"].get("task")

            pass_a_results[query] = actual_task

            if actual_task == expected_task.value:
                correct_routes += 1

        accuracy_pct = round((correct_routes / total) * 100, 2) if total else 0.0

        # -------------------------------
        # PASS B — determinism check
        # -------------------------------
        stable_decisions = 0

        for query, _ in self.TRAFFIC:
            result = self.engine.run(query)
            current_task = result["agent_decisions"].get("task")
            previous_task = pass_a_results.get(query)

            if current_task == previous_task:
                stable_decisions += 1

        determinism_pct = (
            round((stable_decisions / total) * 100, 2) if total else 0.0
        )

        # -------------------------------
        # KPI Assembly
        # -------------------------------
        metrics: List[KPIMetric] = [
            KPIMetric(
                category="Routing & Intent",
                name="Intent Signal Accuracy",
                value=accuracy_pct,
                samples=total,
            ),
            KPIMetric(
                category="Routing & Intent",
                name="Task Routing Accuracy",
                value=accuracy_pct,
                samples=total,
            ),
            KPIMetric(
                category="System Stability",
                name="Routing Determinism Rate",
                value=determinism_pct,
                samples=total,
            ),
        ]

        return metrics

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    @staticmethod
    def print_dashboard(metrics: List[KPIMetric]) -> None:
        print(f"\n{BOLD}{CYAN}RESUME-GRADE KPI REPORT{RESET}\n")

        header = (
            f"{BOLD}| Metric Category | Metric Name             "
            f"| Value        | N (Samples) |{RESET}"
        )
        divider = "-" * len(header)

        print(divider)
        print(header)
        print(divider)

        for metric in metrics:
            value_str = f"{GREEN}{metric.value:.2f}%{RESET}"

            print(
                f"| {metric.category:<15} "
                f"| {metric.name:<23} "
                f"| {value_str:<12} "
                f"| {metric.samples:<11} |"
            )

        print(divider)
        print()


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    dashboard = ResumeKPIDashboard()
    metrics = dashboard.run()
    dashboard.print_dashboard(metrics)
