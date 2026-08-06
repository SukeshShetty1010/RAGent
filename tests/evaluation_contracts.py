# FILE: tests/evaluation_contracts.py
# Metric Result Contracts (PURE DATA ONLY)
#
# Design principles:
# - Immutable (frozen dataclasses)
# - No logic, no imports from engine or retriever
# - Resume-grade metrics are first-class citizens
# - Backward compatible with existing KPIs

from __future__ import annotations
from dataclasses import dataclass


# ============================================================
# Legacy / Diagnostic Metrics (NOT resume-facing)
# ============================================================

@dataclass(frozen=True)
class PrecisionAtKResult:
    """
    Diagnostic-only metric.
    Kept for internal debugging, not resume-facing.
    """
    precision: float
    hits: int
    k: int


# ============================================================
# Resume-Grade Retrieval Metrics (FIX 1 + FIX 2)
# ============================================================

@dataclass(frozen=True)
class EvidenceHitRateResult:
    """
    Measures whether at least one expected entity/source
    was retrieved within top-K results.
    """
    hit_rate: float
    hit_queries: int
    total_queries: int


@dataclass(frozen=True)
class EntityCoverageResult:
    """
    Measures how many expected entities were covered
    by retrieved evidence (important for comparisons).
    """
    coverage: float
    covered_entities: int
    expected_entities: int


# ============================================================
# Grounding & Hallucination Control
# ============================================================

@dataclass(frozen=True)
class GroundingFidelityResult:
    fidelity: float
    grounded_sentences: int
    total_sentences: int


@dataclass(frozen=True)
class HallucinationAvoidanceResult:
    """
    Complement of grounding failures.
    """
    avoidance_rate: float
    ungrounded_sentences: int
    total_sentences: int


# ============================================================
# Context & Efficiency Metrics
# ============================================================

@dataclass(frozen=True)
class CompressionRatioResult:
    ratio: float
    initial_count: int
    final_count: int


# ============================================================
# Stability & Regression Metrics
# ============================================================

@dataclass(frozen=True)
class StabilityRateResult:
    stability_rate: float
    passed_runs: int
    total_runs: int


# ============================================================
# Honesty & Capability Metrics (CORE DIFFERENTIATOR)
# ============================================================

@dataclass(frozen=True)
class HonestyRateResult:
    """
    FULL + PARTIAL answers counted as honest behavior.
    """
    honesty_rate: float
    honest_runs: int
    total_runs: int


@dataclass(frozen=True)
class CapabilityDistributionResult:
    """
    Distribution of FULL / PARTIAL / INSUFFICIENT answers.
    """
    full_rate: float
    partial_rate: float
    insufficient_rate: float
    total_runs: int
