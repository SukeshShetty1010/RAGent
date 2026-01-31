# ============================================================
# tests/evaluation_metrics.py
# Deterministic Scoring Engine for RAG Evaluation (CAPABILITY-AWARE)
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any


# ============================================================
# Metric Result Contracts
# ============================================================

@dataclass(frozen=True)
class PrecisionAtKResult:
    precision: float
    hits: int
    k: int


@dataclass(frozen=True)
class GroundingFidelityResult:
    fidelity: float
    grounded_sentences: int
    total_sentences: int


@dataclass(frozen=True)
class HallucinationAvoidanceResult:
    avoidance_rate: float
    ungrounded_sentences: int
    total_sentences: int


@dataclass(frozen=True)
class CompressionRatioResult:
    ratio: float
    initial_count: int
    final_count: int


@dataclass(frozen=True)
class StabilityRateResult:
    stability_rate: float
    passed_runs: int
    total_runs: int


@dataclass(frozen=True)
class HonestyRateResult:
    honesty_rate: float
    honest_runs: int
    total_runs: int


@dataclass(frozen=True)
class CapabilityDistributionResult:
    full_rate: float
    partial_rate: float
    insufficient_rate: float
    total_runs: int


# ============================================================
# Internal Utilities
# ============================================================

def _normalize_title(title: str) -> str:
    """
    Deterministic normalization for source titles.
    """
    return title.lower().strip()


# ============================================================
# Metric 1: Retrieval Precision @ K
# ============================================================

def calculate_precision_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_source_titles: List[str],
    k: int,
) -> PrecisionAtKResult:

    if k <= 0:
        return PrecisionAtKResult(0.0, 0, k)

    if not retrieved_chunks or not expected_source_titles:
        return PrecisionAtKResult(0.0, 0, k)

    top_k = retrieved_chunks[:k]
    expected = {_normalize_title(t) for t in expected_source_titles}

    hits = sum(
        1
        for c in top_k
        if _normalize_title(c.get("source_title", "")) in expected
    )

    precision = round(hits / float(k), 4)
    return PrecisionAtKResult(precision, hits, k)


# ============================================================
# Metric 2: Grounding Fidelity
# ============================================================

def calculate_grounding_fidelity(
    answer_text: str,
    context_chunks: List[Dict[str, Any]],
) -> GroundingFidelityResult:

    if not answer_text:
        return GroundingFidelityResult(0.0, 0, 0)

    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", answer_text)
        if s.strip()
    ]

    if not sentences:
        return GroundingFidelityResult(0.0, 0, 0)

    context_titles = {
        _normalize_title(c.get("source_title", ""))
        for c in context_chunks
        if c.get("source_title")
    }

    citation_pattern = re.compile(r"\(Source:\s*'([^']+)'\)")
    grounded = 0

    for sentence in sentences:
        match = citation_pattern.search(sentence)
        if match:
            cited = _normalize_title(match.group(1))
            if cited in context_titles:
                grounded += 1

    fidelity = round(grounded / float(len(sentences)), 4)

    return GroundingFidelityResult(
        fidelity=fidelity,
        grounded_sentences=grounded,
        total_sentences=len(sentences),
    )


# ============================================================
# Metric B2: Hallucination Avoidance Rate
# ============================================================

def calculate_hallucination_avoidance(
    grounding: GroundingFidelityResult,
) -> HallucinationAvoidanceResult:

    total = grounding.total_sentences
    if total == 0:
        return HallucinationAvoidanceResult(0.0, 0, 0)

    ungrounded = total - grounding.grounded_sentences
    avoidance = round(1.0 - (ungrounded / float(total)), 4)

    return HallucinationAvoidanceResult(
        avoidance_rate=avoidance,
        ungrounded_sentences=ungrounded,
        total_sentences=total,
    )


# ============================================================
# Metric 3: Context Compression Ratio
# ============================================================

def calculate_compression_ratio(
    initial_retrieved_count: int,
    final_assembled_count: int,
) -> CompressionRatioResult:

    if initial_retrieved_count <= 0:
        return CompressionRatioResult(
            0.0,
            initial_retrieved_count,
            final_assembled_count,
        )

    ratio = round(
        final_assembled_count / float(initial_retrieved_count),
        4,
    )

    return CompressionRatioResult(
        ratio=ratio,
        initial_count=initial_retrieved_count,
        final_count=final_assembled_count,
    )


# ============================================================
# Metric 4: Latency Breakdown
# ============================================================

def analyze_latency_profile(
    metrics_dump: Dict[str, Any],
) -> Dict[str, float]:

    if not metrics_dump or "distributions" not in metrics_dump:
        return {}

    latency_metrics: Dict[str, float] = {}

    for name, payload in metrics_dump["distributions"].items():
        if (
            name.startswith("latency::")
            and isinstance(payload.get("avg"), (int, float))
        ):
            latency_metrics[name] = round(
                float(payload["avg"]), 4
            )

    return latency_metrics


# ============================================================
# Metric 5: Regression Stability Rate
# ============================================================

def calculate_stability_rate(
    total_runs: int,
    passed_runs: int,
) -> StabilityRateResult:

    if total_runs <= 0:
        return StabilityRateResult(0.0, passed_runs, total_runs)

    rate = round(passed_runs / float(total_runs), 4)

    return StabilityRateResult(
        stability_rate=rate,
        passed_runs=passed_runs,
        total_runs=total_runs,
    )


# ============================================================
# Metric 6: Capability Distribution
# ============================================================

def calculate_capability_distribution(
    execution_results: List[Dict[str, Any]],
) -> CapabilityDistributionResult:
    """
    Measures how often the system answers FULL / PARTIAL / INSUFFICIENT.
    """

    total = len(execution_results)
    if total == 0:
        return CapabilityDistributionResult(0.0, 0.0, 0.0, 0)

    full = 0
    partial = 0
    insufficient = 0

    for r in execution_results:
        cap = (
            r.get("kpis", {}) or {}
        ).get("answer_capability")

        if cap == "full":
            full += 1
        elif cap == "partial":
            partial += 1
        elif cap == "insufficient":
            insufficient += 1

    return CapabilityDistributionResult(
        full_rate=round(full / total, 4),
        partial_rate=round(partial / total, 4),
        insufficient_rate=round(insufficient / total, 4),
        total_runs=total,
    )


# ============================================================
# Metric B1: Honesty Rate
# ============================================================

def calculate_honesty_rate(
    execution_results: List[Dict[str, Any]],
) -> HonestyRateResult:
    """
    Aggregates 'full' and 'partial' capabilities as 'honest'.
    """

    total = len(execution_results)
    if total == 0:
        return HonestyRateResult(0.0, 0, 0)

    honest = 0
    for r in execution_results:
        cap = (r.get("kpis", {}) or {}).get("answer_capability")
        if cap in ("full", "partial"):
            honest += 1

    return HonestyRateResult(
        honesty_rate=round(honest / float(total), 4),
        honest_runs=honest,
        total_runs=total,
    )