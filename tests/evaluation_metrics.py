# ============================================================
# tests/evaluation_metrics.py
# Deterministic Scoring Engine for RAG Evaluation (FINAL)
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
class CompressionRatioResult:
    ratio: float
    initial_count: int
    final_count: int


@dataclass(frozen=True)
class StabilityRateResult:
    stability_rate: float
    passed_runs: int
    total_runs: int


# ============================================================
# Internal Utilities (Deterministic Normalization)
# ============================================================

def _normalize_title(title: str) -> str:
    """
    Deterministic normalization for source titles.
    No fuzzy matching, no NLP.
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
    """
    Precision@K = (# of relevant chunks in top K) / K
    """
    if k <= 0:
        return PrecisionAtKResult(0.0, 0, k)

    if not retrieved_chunks or not expected_source_titles:
        return PrecisionAtKResult(0.0, 0, k)

    top_k = retrieved_chunks[:k]

    expected_set = {
        _normalize_title(t) for t in expected_source_titles
    }

    hits = sum(
        1
        for c in top_k
        if _normalize_title(c.get("source_title", "")) in expected_set
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
    """
    Grounding Fidelity = grounded_sentences / total_sentences

    A sentence is grounded ONLY if:
    - It contains a (Source: 'Title') citation
    - The cited title exists in the provided context
    """
    if not answer_text or not isinstance(answer_text, str):
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
            cited_title = _normalize_title(match.group(1))
            if cited_title in context_titles:
                grounded += 1

    fidelity = round(grounded / float(len(sentences)), 4)

    return GroundingFidelityResult(
        fidelity=fidelity,
        grounded_sentences=grounded,
        total_sentences=len(sentences),
    )


# ============================================================
# Metric 3: Context Compression Ratio
# ============================================================

def calculate_compression_ratio(
    initial_retrieved_count: int,
    final_assembled_count: int,
) -> CompressionRatioResult:
    """
    Compression Ratio = final / initial
    """
    if initial_retrieved_count <= 0:
        return CompressionRatioResult(
            0.0, initial_retrieved_count, final_assembled_count
        )

    ratio = round(
        final_assembled_count / float(initial_retrieved_count), 4
    )

    return CompressionRatioResult(
        ratio=ratio,
        initial_count=initial_retrieved_count,
        final_count=final_assembled_count,
    )


# ============================================================
# Metric 4: Latency Breakdown
# ============================================================

def analyze_latency_profile(metrics_dump: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract and normalize latency:: metrics from MetricsRegistry output.
    """
    if not metrics_dump or "distributions" not in metrics_dump:
        return {}

    latency_metrics: Dict[str, float] = {}

    for name, payload in metrics_dump["distributions"].items():
        if name.startswith("latency::") and isinstance(payload.get("avg"), (int, float)):
            latency_metrics[name] = round(float(payload["avg"]), 4)

    return latency_metrics


# ============================================================
# Metric 5: Regression Stability Rate
# ============================================================

def calculate_stability_rate(
    total_runs: int,
    passed_runs: int,
) -> StabilityRateResult:
    """
    Stability Rate = passed_runs / total_runs
    """
    if total_runs <= 0:
        return StabilityRateResult(0.0, passed_runs, total_runs)

    rate = round(passed_runs / float(total_runs), 4)

    return StabilityRateResult(
        stability_rate=rate,
        passed_runs=passed_runs,
        total_runs=total_runs,
    )



# ============================================================
# Mock Usage Examples
# ============================================================

if __name__ == "__main__":
    # ------------------ Precision@K ------------------
    retrieved = [
        {"source_title": "Doc A"},
        {"source_title": "Doc B"},
        {"source_title": "Doc C"},
    ]

    precision = calculate_precision_at_k(
        retrieved_chunks=retrieved,
        expected_source_titles=["Doc A", "Doc C"],
        k=2,
    )
    print("Precision@K:", precision)

    # ---------------- Grounding Fidelity --------------
    answer = (
        "Far Cry 5 was released in 2018 (Source: 'Doc A'). "
        "It was developed by Ubisoft (Source: 'Fake Book')."
    )

    context = [
        {"source_title": "Doc A"},
        {"source_title": "Doc B"},
    ]

    grounding = calculate_grounding_fidelity(answer, context)
    print("Grounding Fidelity:", grounding)

    # ---------------- Compression Ratio ---------------
    compression = calculate_compression_ratio(20, 8)
    print("Compression Ratio:", compression)

    # ---------------- Latency Profile -----------------
    mock_metrics = {
        "distributions": {
            "latency::REQUEST_TOTAL": {"avg": 120.56789},
            "latency::Retrieval": {"avg": 45.12345},
        }
    }

    latency = analyze_latency_profile(mock_metrics)
    print("Latency Profile:", latency)

    # ---------------- Stability Rate ------------------
    stability = calculate_stability_rate(total_runs=50, passed_runs=47)
    print("Stability Rate:", stability)
