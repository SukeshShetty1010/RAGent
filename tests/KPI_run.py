# ============================================================
# tests/KPI_run.py
# Executive KPI Aggregation Harness (FINAL + KPI 8)
# ============================================================

from __future__ import annotations

import time
import math
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

from agent.task_router import TaskType
from tests.observability import MetricsRegistry
from tests.e2e_run import E2EProbe
from tests.regression_suite import RegressionRunner


# ------------------------------------------------------------
# ANSI Colors (Dashboard)
# ------------------------------------------------------------

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


# ------------------------------------------------------------
# KPI Report Contracts
# ------------------------------------------------------------

@dataclass
class LatencyKPI:
    p50_ms: float
    p95_ms: float


@dataclass
class CacheKPI:
    cold_ms: float
    warm_ms: float
    speedup: float


@dataclass
class QualityKPI:
    quality_ok_pct: float
    quality_weak_pct: float
    quality_empty_pct: float


@dataclass
class ContextKPI:
    noise_reduction_ratio: float


@dataclass
class RegressionKPI:
    stability_rate: float


@dataclass
class TaskRoutingKPI:
    accuracy_pct: float
    total_samples: int
    correct_samples: int


@dataclass
class TaskCompletionKPI:
    success_rate_pct: float
    total_attempts: int
    successful_attempts: int


@dataclass
class ConfidenceKPI:
    avg_score: float
    min_score: float
    max_score: float
    p95_score: float


@dataclass
class KPIReport:
    latency: LatencyKPI
    cache: CacheKPI
    retrieval_quality: QualityKPI
    context_efficiency: ContextKPI
    regression: RegressionKPI
    task_routing: TaskRoutingKPI
    task_completion: TaskCompletionKPI
    confidence: ConfidenceKPI


# ============================================================
# KPI Suite
# ============================================================

class KPISuite:
    """
    Executive KPI aggregation harness.
    """

    # --------------------------------------------------------
    # Percentile Helper (NO numpy)
    # --------------------------------------------------------

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = math.ceil((pct / 100.0) * len(values)) - 1
        k = max(0, min(k, len(values) - 1))
        return round(values[k], 4)

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run(self) -> KPIReport:
        logging.getLogger().setLevel(logging.WARNING)

        registry = MetricsRegistry.get()
        probe = E2EProbe()

        # =====================================================
        # KPI 2 — Cache Speedup (ISOLATED FIRST)
        # =====================================================

        COLD_QUERY = "History of AI architectures"

        start = time.perf_counter()
        probe.run_pipeline(COLD_QUERY)
        cold_time = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        probe.run_pipeline(COLD_QUERY)
        warm_time = (time.perf_counter() - start) * 1000.0

        speedup = round(cold_time / warm_time, 2) if warm_time > 0 else 0.0

        cache_kpi = CacheKPI(
            cold_ms=round(cold_time, 2),
            warm_ms=round(warm_time, 2),
            speedup=speedup,
        )

        # =====================================================
        # Ground-Truth Traffic Sweep (KPI 3 + 6 + 7 + 8)
        # =====================================================

        TRAFFIC_SWEEP: List[Tuple[str, TaskType]] = [
            ("Compare Assassin's Creed Valhalla vs Far Cry 5", TaskType.COMPARISON),
            ("Top 10 things to do in Far Cry 5", TaskType.LISTICLE),
            ("What is the release date of Far Cry 5?", TaskType.FACTUAL),
            ("Explain why Far Cry 5 is controversial", TaskType.OPEN),
            ("Latest patch notes for Assassin's Creed Valhalla", TaskType.OPEN),
        ]

        quality_counts: Dict[str, int] = {
            "quality_ok": 0,
            "quality_weak": 0,
            "quality_empty": 0,
        }

        routing_correct = 0
        total_attempts = 0
        successful_attempts = 0
        confidence_scores: List[float] = []

        for query, expected_task in TRAFFIC_SWEEP:
            total_attempts += 1

            try:
                result = probe.run_pipeline(query)
            except Exception as exc:
                logging.error(f"KPI-7 failure (exception): {exc}")
                continue  # fail-soft

            behavior = result.get("behavior", {})
            quality_status = behavior.get("quality_status")
            output_preview = result.get("output_preview")
            confidence = behavior.get("confidence_score")

            # ---- KPI 3: Retrieval Quality ----
            if quality_status in quality_counts:
                quality_counts[quality_status] += 1

            # ---- KPI 6: Task Routing Accuracy ----
            if behavior.get("task_type") == expected_task.value:
                routing_correct += 1

            # ---- KPI 7: Task Completion Success ----
            if (
                quality_status != "quality_empty"
                and isinstance(output_preview, str)
                and output_preview.strip()
            ):
                successful_attempts += 1

            # ---- KPI 8: Confidence Distribution ----
            if isinstance(confidence, (int, float)):
                confidence_scores.append(float(confidence))

        total_runs = len(TRAFFIC_SWEEP) or 1

        quality_kpi = QualityKPI(
            quality_ok_pct=round((quality_counts["quality_ok"] / total_runs) * 100.0, 2),
            quality_weak_pct=round((quality_counts["quality_weak"] / total_runs) * 100.0, 2),
            quality_empty_pct=round((quality_counts["quality_empty"] / total_runs) * 100.0, 2),
        )

        routing_kpi = TaskRoutingKPI(
            accuracy_pct=round((routing_correct / total_runs) * 100.0, 2),
            total_samples=total_runs,
            correct_samples=routing_correct,
        )

        completion_kpi = TaskCompletionKPI(
            success_rate_pct=round((successful_attempts / total_attempts) * 100.0, 2)
            if total_attempts > 0 else 0.0,
            total_attempts=total_attempts,
            successful_attempts=successful_attempts,
        )

        confidence_kpi = ConfidenceKPI(
            avg_score=round(sum(confidence_scores) / len(confidence_scores), 4)
            if confidence_scores else 0.0,
            min_score=round(min(confidence_scores), 4) if confidence_scores else 0.0,
            max_score=round(max(confidence_scores), 4) if confidence_scores else 0.0,
            p95_score=self._percentile(confidence_scores, 95),
        )

        # =====================================================
        # KPI 1 — Latency
        # =====================================================

        latency_values = registry._distributions.get(
            "latency::REQUEST_TOTAL"
        ).values

        latency_kpi = LatencyKPI(
            p50_ms=self._percentile(latency_values, 50),
            p95_ms=self._percentile(latency_values, 95),
        )

        # =====================================================
        # KPI 4 — Context Noise Reduction
        # =====================================================

        input_chunks = sum(
            registry._distributions.get("context_input_chunks").values
        )
        final_chunks = sum(
            registry._distributions.get("context_final_chunks").values
        )

        reduction = 1.0 - (final_chunks / input_chunks) if input_chunks > 0 else 0.0

        context_kpi = ContextKPI(noise_reduction_ratio=round(reduction, 4))

        # =====================================================
        # KPI 5 — Regression Stability
        # =====================================================

        regression_passed = RegressionRunner().run()
        regression_kpi = RegressionKPI(
            stability_rate=1.0 if regression_passed else 0.0
        )

        # =====================================================
        # Final Report
        # =====================================================

        report = KPIReport(
            latency=latency_kpi,
            cache=cache_kpi,
            retrieval_quality=quality_kpi,
            context_efficiency=context_kpi,
            regression=regression_kpi,
            task_routing=routing_kpi,
            task_completion=completion_kpi,
            confidence=confidence_kpi,
        )

        self._print_dashboard(report)
        self._print_json(report)
        return report

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    @staticmethod
    def _print_dashboard(report: KPIReport) -> None:
        print(f"\n{CYAN}=== RAGent Executive KPI Dashboard ==={RESET}\n")

        print(f"{GREEN}Latency (ms){RESET}")
        print(f"  P50: {report.latency.p50_ms}")
        print(f"  P95: {report.latency.p95_ms}\n")

        print(f"{GREEN}Cache Performance{RESET}")
        print(f"  Cold: {report.cache.cold_ms} ms")
        print(f"  Warm: {report.cache.warm_ms} ms")
        print(f"  Speedup: {report.cache.speedup}×\n")

        print(f"{GREEN}Retrieval Quality Distribution{RESET}")
        print(f"  QUALITY_OK: {report.retrieval_quality.quality_ok_pct}%")
        print(f"  QUALITY_WEAK: {report.retrieval_quality.quality_weak_pct}%")
        print(f"  QUALITY_EMPTY: {report.retrieval_quality.quality_empty_pct}%\n")

        print(f"{GREEN}Context Efficiency{RESET}")
        print(f"  Noise Reduction: {report.context_efficiency.noise_reduction_ratio * 100:.2f}%\n")

        print(f"{GREEN}Regression Stability{RESET}")
        print(f"  Stability Rate: {report.regression.stability_rate * 100}%\n")

        print(f"{CYAN}🚦 Task Routing Accuracy{RESET}")
        print(
            f"  Accuracy: {report.task_routing.accuracy_pct}% "
            f"({report.task_routing.correct_samples}/{report.task_routing.total_samples})\n"
        )

        print(f"{GREEN}✅ Task Completion Success{RESET}")
        print(
            f"  Success Rate: {report.task_completion.success_rate_pct}% "
            f"({report.task_completion.successful_attempts}/{report.task_completion.total_attempts})\n"
        )

        print(f"{CYAN}🧠 Confidence Distribution{RESET}")
        print(f"  Avg: {report.confidence.avg_score}")
        print(f"  Min: {report.confidence.min_score}")
        print(f"  Max: {report.confidence.max_score}")
        print(f"  P95: {report.confidence.p95_score}\n")

    @staticmethod
    def _print_json(report: KPIReport) -> None:
        print(f"{CYAN}=== KPI JSON SUMMARY ==={RESET}")
        print(json.dumps(asdict(report), indent=2))


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    KPISuite().run()
