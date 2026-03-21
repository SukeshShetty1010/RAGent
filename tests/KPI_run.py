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
from typing import List, Dict, Tuple, Any

from engine.execution_engine import RageEngine
from agent.task_router import TaskType
from agent.capability.capability_types import AnswerCapability

from utils.observability import MetricsRegistry
from tests.regression_suite import RegressionRunner

from tests.evaluation_metrics import (
    calculate_grounding_fidelity,
    calculate_honesty_rate,
)

# ------------------------------------------------------------
# ANSI Colors (CLI Dashboard)
# ------------------------------------------------------------

CYAN = "\033[96m"
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
class HonestyKPI:
    honesty_rate_pct: float
    honest_answers: int
    total_answers: int


@dataclass
class HallucinationAvoidanceKPI:
    avoidance_rate_pct: float
    ungrounded_sentences: int
    total_sentences: int


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
    honesty: HonestyKPI
    hallucination_avoidance: HallucinationAvoidanceKPI


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

        cache_kpi = CacheKPI(
            cold_ms=round(cold_ms, 2),
            warm_ms=round(warm_ms, 2),
            speedup=round(cold_ms / warm_ms, 2) if warm_ms else 0.0,
        )

        # =====================================================
        # Traffic Sweep
        # =====================================================

        TRAFFIC: List[Tuple[str, TaskType]] = [
            ("Compare Far Cry 5 vs Assassin’s Creed Valhalla", TaskType.COMPARISON),
            ("Top 5 things to do in Far Cry 5", TaskType.LISTICLE),
            ("What is the release date of Far Cry 5?", TaskType.FACTUAL),
            ("Explain why Far Cry 5 is controversial", TaskType.OPEN),
            ("Latest update for Assassin’s Creed Valhalla", TaskType.OPEN),
        ]

        quality_counts = {
            "QUALITY_OK": 0,
            "QUALITY_WEAK": 0,
            "QUALITY_EMPTY": 0,
        }

        routing_correct = 0
        total_attempts = 0
        successful_attempts = 0
        confidence_scores: List[float] = []

        capability_counts = {
            "full": 0,
            "partial": 0,
            "insufficient": 0,
        }

        execution_results: List[Dict[str, Any]] = []

        for query, expected_task in TRAFFIC:
            total_attempts += 1
            result = self.engine.run(query)
            execution_results.append(result)

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

            # ---- Retrieval Quality (FIXED) ----
            qs = kpis.get("quality_status", "")
            qs = qs.upper() if isinstance(qs, str) else qs
            if qs in quality_counts:
                quality_counts[qs] += 1

            # ---- Capability ----
            cap = agent.get("answer_capability")
            if cap in capability_counts:
                capability_counts[cap] += 1

        runs = len(TRAFFIC)

        quality_kpi = QualityKPI(
            quality_ok_pct=round(quality_counts["QUALITY_OK"] / runs * 100, 2),
            quality_weak_pct=round(quality_counts["QUALITY_WEAK"] / runs * 100, 2),
            quality_empty_pct=round(quality_counts["QUALITY_EMPTY"] / runs * 100, 2),
        )

        routing_kpi = TaskRoutingKPI(
            accuracy_pct=round(routing_correct / runs * 100, 2),
            total_samples=runs,
            correct_samples=routing_correct,
        )

        completion_kpi = TaskCompletionKPI(
            success_rate_pct=round(successful_attempts / total_attempts * 100, 2),
            total_attempts=total_attempts,
            successful_attempts=successful_attempts,
        )

        confidence_kpi = ConfidenceKPI(
            avg_score=round(sum(confidence_scores) / len(confidence_scores), 4),
            min_score=min(confidence_scores),
            max_score=max(confidence_scores),
            p95_score=self._percentile(confidence_scores, 95),
        )

        capability_kpi = CapabilityKPI(
            full_pct=round(capability_counts["full"] / runs * 100, 2),
            partial_pct=round(capability_counts["partial"] / runs * 100, 2),
            insufficient_pct=round(capability_counts["insufficient"] / runs * 100, 2),
        )

        # =====================================================
        # B1 — Honesty Rate
        # =====================================================

        honesty = calculate_honesty_rate(execution_results)

        honesty_kpi = HonestyKPI(
            honesty_rate_pct=round(honesty.honesty_rate * 100, 2),
            honest_answers=honesty.honest_runs,
            total_answers=honesty.total_runs,
        )

        # =====================================================
        # B2 — Hallucination Avoidance
        # =====================================================

        grounded = 0
        total_sentences = 0

        for res in execution_results:
            gf = calculate_grounding_fidelity(
                answer_text=res.get("final_answer", ""),
                context_chunks=res.get("evidence", []),
            )
            grounded += gf.grounded_sentences
            total_sentences += gf.total_sentences

        hallucination_kpi = HallucinationAvoidanceKPI(
            avoidance_rate_pct=round(
                (grounded / total_sentences * 100) if total_sentences else 0.0,
                2,
            ),
            ungrounded_sentences=total_sentences - grounded,
            total_sentences=total_sentences,
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

        context_kpi = ContextKPI(
            noise_reduction_ratio=round(
                1.0 - (out_chunks / in_chunks) if in_chunks else 0.0,
                4,
            )
        )

        # =====================================================
        # Regression Stability
        # =====================================================

        regression_kpi = RegressionKPI(
            stability_rate=1.0 if RegressionRunner().run() else 0.0
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
            honesty=honesty_kpi,
            hallucination_avoidance=hallucination_kpi,
        )

        print(f"\n{CYAN}=== RAGent KPI Dashboard ==={RESET}\n")
        print(json.dumps(asdict(report), indent=2))

        # Clean shutdown
        self.engine.close()

        return report


if __name__ == "__main__":
    KPISuite().run()
