# ============================================================
# tests/evaluation_metrics.py
# Deterministic Scoring Engine for RAG Evaluation (CAPABILITY-AWARE)
#
# FINAL CANONICAL VERSION (BACKWARD-COMPATIBLE)
# ============================================================

from __future__ import annotations

import re
from typing import List, Dict, Any, Set

from tests.evaluation_contracts import (
    PrecisionAtKResult,
    EvidenceHitRateResult,
    EntityCoverageResult,
    GroundingFidelityResult,
    HallucinationAvoidanceResult,
    CompressionRatioResult,
    StabilityRateResult,
    HonestyRateResult,
    CapabilityDistributionResult,
)

# ============================================================
# Internal Utilities
# ============================================================

def _normalize(text: str) -> str:
    return text.lower().strip()


def _resolve_entity(chunk: Dict[str, Any]) -> str | None:
    for key in (
        "game_name",
        "canonical_game",
        "retrieval_context",
        "source_title",
    ):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize(value)
    return None


def _extract_retrieved_entities(
    retrieved_chunks: List[Dict[str, Any]],
) -> Set[str]:
    entities: Set[str] = set()
    for c in retrieved_chunks:
        entity = _resolve_entity(c)
        if entity:
            entities.add(entity)
    return entities


def _entity_match(retrieved: str, expected: str) -> bool:
    return (
        retrieved == expected
        or expected in retrieved
        or retrieved in expected
    )

# ============================================================
# Diagnostic Metric — Precision @ K
# ============================================================

def calculate_precision_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_source_titles: List[str],
    k: int,
) -> PrecisionAtKResult:

    if k <= 0 or not retrieved_chunks or not expected_source_titles:
        return PrecisionAtKResult(0.0, 0, k)

    expected = {_normalize(t) for t in expected_source_titles}
    hits = 0

    for c in retrieved_chunks[:k]:
        title = _normalize(c.get("source_title", ""))
        if title in expected:
            hits += 1

    return PrecisionAtKResult(round(hits / float(k), 4), hits, k)

# ============================================================
# Evidence Hit Rate
# ============================================================

def calculate_evidence_hit_rate(
    retrieved_chunks: List[Dict[str, Any]],
    expected_entities: List[str],
) -> EvidenceHitRateResult:

    total = 1
    if not retrieved_chunks or not expected_entities:
        return EvidenceHitRateResult(0.0, 0, total)

    retrieved_entities = _extract_retrieved_entities(retrieved_chunks)
    expected = [_normalize(e) for e in expected_entities]

    hit = 0
    for r in retrieved_entities:
        for e in expected:
            if _entity_match(r, e):
                hit = 1
                break
        if hit:
            break

    return EvidenceHitRateResult(
        round(hit / total, 4),
        hit,
        total,
    )

# ============================================================
# Entity Coverage
# ============================================================

def calculate_entity_coverage(
    retrieved_chunks: List[Dict[str, Any]],
    expected_entities: List[str],
) -> EntityCoverageResult:

    if not expected_entities:
        return EntityCoverageResult(0.0, 0, 0)

    retrieved_entities = _extract_retrieved_entities(retrieved_chunks)
    expected = [_normalize(e) for e in expected_entities]

    covered = 0
    for e in expected:
        for r in retrieved_entities:
            if _entity_match(r, e):
                covered += 1
                break

    total = len(expected)

    return EntityCoverageResult(
        round(covered / float(total), 4),
        covered,
        total,
    )

# ============================================================
# Grounding & Hallucination
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
        _normalize(c.get("source_title", ""))
        for c in context_chunks
        if c.get("source_title")
    }

    citation_pattern = re.compile(r"\(Source:\s*'([^']+)'\)")
    grounded = 0

    for sentence in sentences:
        match = citation_pattern.search(sentence)
        if match:
            cited = _normalize(match.group(1))
            if cited in context_titles:
                grounded += 1

    return GroundingFidelityResult(
        round(grounded / float(len(sentences)), 4),
        grounded,
        len(sentences),
    )


def calculate_hallucination_avoidance(
    grounding: GroundingFidelityResult,
) -> HallucinationAvoidanceResult:

    total = grounding.total_sentences
    if total == 0:
        return HallucinationAvoidanceResult(0.0, 0, 0)

    ungrounded = total - grounding.grounded_sentences

    return HallucinationAvoidanceResult(
        round(1.0 - (ungrounded / float(total)), 4),
        ungrounded,
        total,
    )

# ============================================================
# ✅ FIX — Context Compression as Noise Reduction
# ============================================================

def calculate_compression_ratio(
    input_chunks: int,
    final_chunks: int,
) -> CompressionRatioResult:
    """
    Resume-grade interpretation:
    ratio = % of context eliminated (noise removed)
    """

    if input_chunks <= 0:
        return CompressionRatioResult(0.0, input_chunks, final_chunks)

    removed = input_chunks - final_chunks
    ratio = max(0.0, removed / float(input_chunks))

    return CompressionRatioResult(
        round(ratio, 4),
        input_chunks,
        final_chunks,
    )

# ============================================================
# Latency Analysis
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
            latency_metrics[name] = round(float(payload["avg"]), 4)

    return latency_metrics

# ============================================================
# Stability, Honesty & Capability Metrics
# ============================================================

def calculate_stability_rate(
    total_runs: int,
    passed_runs: int,
) -> StabilityRateResult:

    if total_runs <= 0:
        return StabilityRateResult(0.0, passed_runs, total_runs)

    return StabilityRateResult(
        round(passed_runs / float(total_runs), 4),
        passed_runs,
        total_runs,
    )


def calculate_capability_distribution(
    execution_results: List[Dict[str, Any]],
) -> CapabilityDistributionResult:

    total = len(execution_results)
    if total == 0:
        return CapabilityDistributionResult(0.0, 0.0, 0.0, 0)

    full = partial = insufficient = 0

    for r in execution_results:
        cap = (r.get("kpis", {}) or {}).get("answer_capability")
        if cap == "full":
            full += 1
        elif cap == "partial":
            partial += 1
        elif cap == "insufficient":
            insufficient += 1

    return CapabilityDistributionResult(
        round(full / total, 4),
        round(partial / total, 4),
        round(insufficient / total, 4),
        total,
    )

# ============================================================
# ✅ HONESTY RATE (KEY RESUME KPI)
# ============================================================

def calculate_honesty_rate(
    execution_results: List[Dict[str, Any]],
) -> HonestyRateResult:
    """
    Honest answers = FULL + PARTIAL
    Dishonest answers = hallucination / unsafe output
    """

    total = len(execution_results)
    if total == 0:
        return HonestyRateResult(0.0, 0, 0)

    honest = sum(
        1
        for r in execution_results
        if (r.get("kpis", {}) or {}).get("answer_capability")
        in ("full", "partial")
    )

    return HonestyRateResult(
        round(honest / float(total), 4),
        honest,
        total,
    )
