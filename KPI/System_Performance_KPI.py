# ============================================================
# tests/KPI/System_Performance_KPI.py
# RESUME-GRADE KPI: SYSTEM PERFORMANCE
# ============================================================

from __future__ import annotations

import logging
import math
from typing import List, Tuple

from engine.execution_engine import RageEngine
from agent.task_router import TaskType
from tests.observability import MetricsRegistry
from tests.regression_suite import RegressionRunner, REGRESSION_VAULT

# ------------------------------------------------------------
# ANSI formatting (resume-grade CLI table)
# ------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"


# ============================================================
# Standard Traffic (copied from KPI_run.py)
# ============================================================

TRAFFIC: List[Tuple[str, TaskType]] = [
    ("Compare Far Cry 5 vs Assassin’s Creed Valhalla", TaskType.COMPARISON),
    ("Top 5 things to do in Far Cry 5", TaskType.LISTICLE),
    ("What is the release date of Far Cry 5?", TaskType.FACTUAL),
    ("Explain why Far Cry 5 is controversial", TaskType.OPEN),
    ("Latest update for Assassin’s Creed Valhalla", TaskType.OPEN),
]


# ============================================================
# Percentile Helper (exact logic from KPI_run.py)
# ============================================================

def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = math.ceil((pct / 100.0) * len(values)) - 1
    k = max(0, min(k, len(values) - 1))
    return round(values[k], 4)


# ============================================================
# KPI Runner
# ============================================================

class SystemPerformanceKPI:
    """
    Targeted KPI runner for System Performance.

    Measures:
    - End-to-End Latency p50 (Median)
    - End-to-End Latency p95 (Tail)
    - LLM Invocation Rate
    - Cache Speedup Factor (Cold vs Warm)
    - Regression Stability Rate

    Purpose:
    - Prove system speed & SLO compliance
    - Demonstrate cost-awareness, caching efficiency, and stability
    """

    def __init__(self) -> None:
        self.engine = RageEngine()
        self.registry = MetricsRegistry.get()

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def run(self) -> None:
        logging.getLogger().setLevel(logging.WARNING)

        # =====================================================
        # PASS 1 — COLD RUN
        # =====================================================

        self.registry._counters.clear()
        self.registry._distributions.clear()
        self.registry._categoricals.clear()

        llm_invocations = 0
        total_requests = len(TRAFFIC)

        for query, _ in TRAFFIC:
            result = self.engine.run(query)
            if result.get("kpis", {}).get("llm_ran"):
                llm_invocations += 1

        cold_latency_metric = self.registry._distributions.get(
            "latency::REQUEST_TOTAL"
        )
        cold_latencies = (
            cold_latency_metric.values if cold_latency_metric else []
        )

        cold_avg_latency = (
            sum(cold_latencies) / len(cold_latencies)
            if cold_latencies else 0.0
        )

        # =====================================================
        # PASS 2 — WARM RUN (CACHE HITS)
        # =====================================================

        self.registry._distributions.clear()

        for query, _ in TRAFFIC:
            self.engine.run(query)

        warm_latency_metric = self.registry._distributions.get(
            "latency::REQUEST_TOTAL"
        )
        warm_latencies = (
            warm_latency_metric.values if warm_latency_metric else []
        )

        warm_avg_latency = (
            sum(warm_latencies) / len(warm_latencies)
            if warm_latencies else 0.0
        )

        # =====================================================
        # REGRESSION STABILITY CHECK
        # =====================================================

        runner = RegressionRunner()
        passed = 0
        total_cases = len(REGRESSION_VAULT)

        for case in REGRESSION_VAULT:
            result = runner._run_case(case)
            if result is None:
                continue
            ok, _ = runner._compare(case.test_case, result)
            if ok:
                passed += 1

        regression_stability_rate = (
            round((passed / total_cases) * 100, 2)
            if total_cases else 0.0
        )

        # =====================================================
        # Metric Calculations
        # =====================================================

        p50 = _percentile(cold_latencies, 50)
        p95 = _percentile(cold_latencies, 95)

        llm_invocation_rate = (
            round((llm_invocations / total_requests) * 100, 2)
            if total_requests else 0.0
        )

        cache_speedup = (
            round(cold_avg_latency / warm_avg_latency, 2)
            if warm_avg_latency > 0 else 0.0
        )

        # =====================================================
        # Dashboard Output
        # =====================================================

        print(
            f"\n{BOLD}{CYAN}RESUME-GRADE KPI: SYSTEM PERFORMANCE{RESET}\n"
        )

        header = (
            f"{BOLD}| Metric Name               "
            f"| Value        | Target        |{RESET}"
        )
        divider = "-" * len(header)

        print(divider)
        print(header)
        print(divider)

        print(
            f"| Latency p50 (Med)         "
            f"| {p50:>7.2f} ms  "
            f"| < 800 ms     |"
        )
        print(
            f"| Latency p95 (Tail)        "
            f"| {p95:>7.2f} ms  "
            f"| < 2000 ms    |"
        )
        print(
            f"| LLM Invocation Rate       "
            f"| {llm_invocation_rate:>6.2f}%     "
            f"| < 80%        |"
        )
        print(
            f"| Cache Speedup Factor      "
            f"| {GREEN}{cache_speedup:>6.2f}x{RESET}     "
            f"| > 10x        |"
        )
        print(
            f"| Regression Stability Rate "
            f"| {GREEN}{regression_stability_rate:>6.2f}%{RESET}     "
            f"| 100%         |"
        )

        print(divider)
        print()

        # Clean shutdown
        self.engine.close()


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    SystemPerformanceKPI().run()
