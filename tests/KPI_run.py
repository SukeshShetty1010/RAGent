# ============================================================
# tests/KPI_run.py
# Executive KPI Aggregation Harness (CAPABILITY-AWARE, FINAL)
# ============================================================

from __future__ import annotations

import time
import math
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

from engine.execution_engine import RageEngine
from agent.task_router import TaskType
from agent.capability.capability_types import AnswerCapability

from tests.observability import MetricsRegistry
from tests.regression_suite import RegressionRunner


# ------------------------------------------------------------
# ANSI Colors (CLI Dashboard)
# ------------------------------------------------------------

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


# ============================================================
# KPI Contracts
# ============================================================

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
class CapabilityKPI:
    full_pct: float
    partial_pct: float
    insufficient_pct: float


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
    capability: CapabilityKPI


# ============================================================
# KPI Suite
# ============================================================

class KPISuite:
    """
    Executive KPI runner aligned with the capability-aware engine.
    """

    def __init__(self) -> None:
        self.engine = RageEngine()
        self.registry = MetricsRegistry.get()

    # --------------------------------------------------------
    # Percentile Helper (no numpy)
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
    # Run KPI Suite
    # --------------------------------------------------------

    def run(self) -> KPIReport:
        logging.getLogger().setLevel(logging.WARNING)

        # =====================================================
        # KPI 1 — Cache Performance
        # =====================================================

        COLD_QUERY = "History of artificial intelligence"

        start = time.perf_counter()
        self.engine.run(COLD_QUERY)
        cold_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        self.engine.run(COLD_QUERY)
        warm_ms = (time.perf_counter() - start) * 1000.0

        speedup = round(cold_ms / warm_ms, 2) if warm_ms > 0 else 0.0

        cache_kpi = CacheKPI(
            cold_ms=round(cold_ms, 2),
            warm_ms=round(warm_ms, 2),
            speedup=speedup,
        )

        # =====================================================
        # Traffic Sweep (Core KPIs)
        # =====================================================

        TRAFFIC: List[Tuple[str, TaskType]] = [
            ("Compare Far Cry 5 vs Assassin’s Creed Valhalla", TaskType.COMPARISON),
            ("Top 5 things to do in Far Cry 5", TaskType.LISTICLE),
            ("What is the release date of Far Cry 5?", TaskType.FACTUAL),
            ("Explain why Far Cry 5 is controversial", TaskType.OPEN),
            ("Latest update for Assassin’s Creed Valhalla", TaskType.OPEN),
        ]

        quality_counts = {"QUALITY_OK": 0, "QUALITY_WEAK": 0, "QUALITY_EMPTY": 0}
        routing_correct = 0
        total_attempts = 0
        successful_attempts = 0
        confidence_scores: List[float] = []

        capability_counts = {
            AnswerCapability.FULL.value: 0,
            AnswerCapability.PARTIAL.value: 0,
            AnswerCapability.INSUFFICIENT.value: 0,
        }

        for query, expected_task in TRAFFIC:
            total_attempts += 1
            result = self.engine.run(query)

            agent = result["agent_decisions"]
            kpis = result["kpis"]

            # ---- Routing Accuracy ----
            if agent.get("task") == expected_task.value:
                routing_correct += 1

            # ---- Task Completion ----
            if kpis.get("llm_ran"):
                successful_attempts += 1

            # ---- Confidence ----
            if isinstance(kpis.get("confidence_score"), (int, float)):
                confidence_scores.append(float(kpis["confidence_score"]))

            # ---- Retrieval Quality ----
            qs = kpis.get("quality_status")
            if qs in quality_counts:
                quality_counts[qs] += 1

            # ---- Capability ----
            cap = agent.get("answer_capability")
            if cap in capability_counts:
                capability_counts[cap] += 1

        runs = len(TRAFFIC)

        quality_kpi = QualityKPI(
            quality_ok_pct=round((quality_counts["QUALITY_OK"] / runs) * 100, 2),
            quality_weak_pct=round((quality_counts["QUALITY_WEAK"] / runs) * 100, 2),
            quality_empty_pct=round((quality_counts["QUALITY_EMPTY"] / runs) * 100, 2),
        )

        routing_kpi = TaskRoutingKPI(
            accuracy_pct=round((routing_correct / runs) * 100, 2),
            total_samples=runs,
            correct_samples=routing_correct,
        )

        completion_kpi = TaskCompletionKPI(
            success_rate_pct=round((successful_attempts / total_attempts) * 100, 2),
            total_attempts=total_attempts,
            successful_attempts=successful_attempts,
        )

        confidence_kpi = ConfidenceKPI(
            avg_score=round(sum(confidence_scores) / len(confidence_scores), 4)
            if confidence_scores else 0.0,
            min_score=min(confidence_scores) if confidence_scores else 0.0,
            max_score=max(confidence_scores) if confidence_scores else 0.0,
            p95_score=self._percentile(confidence_scores, 95),
        )

        capability_kpi = CapabilityKPI(
            full_pct=round((capability_counts["full"] / runs) * 100, 2),
            partial_pct=round((capability_counts["partial"] / runs) * 100, 2),
            insufficient_pct=round((capability_counts["insufficient"] / runs) * 100, 2),
        )

        # =====================================================
        # Latency KPI
        # =====================================================

        latency_vals = (
            self.registry._distributions
            .get("latency::REQUEST_TOTAL", {})
            .values
        )

        latency_kpi = LatencyKPI(
            p50_ms=self._percentile(latency_vals, 50),
            p95_ms=self._percentile(latency_vals, 95),
        )

        # =====================================================
        # Context Efficiency
        # =====================================================

        in_chunks = sum(
            self.registry._distributions.get("context_input_chunks", {}).values
        )
        out_chunks = sum(
            self.registry._distributions.get("context_final_chunks", {}).values
        )

        reduction = 1.0 - (out_chunks / in_chunks) if in_chunks else 0.0
        context_kpi = ContextKPI(noise_reduction_ratio=round(reduction, 4))

        # =====================================================
        # Regression Stability
        # =====================================================

        regression_ok = RegressionRunner().run()
        regression_kpi = RegressionKPI(
            stability_rate=1.0 if regression_ok else 0.0
        )

        report = KPIReport(
            latency=latency_kpi,
            cache=cache_kpi,
            retrieval_quality=quality_kpi,
            context_efficiency=context_kpi,
            regression=regression_kpi,
            task_routing=routing_kpi,
            task_completion=completion_kpi,
            confidence=confidence_kpi,
            capability=capability_kpi,
        )

        self._print(report)
        return report

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    @staticmethod
    def _print(report: KPIReport) -> None:
        print(f"\n{CYAN}=== RAGent KPI Dashboard ==={RESET}\n")
        print(json.dumps(asdict(report), indent=2))


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    KPISuite().run()
