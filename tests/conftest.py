"""
tests/conftest.py

Shared fixtures for hermetic unit tests. Fixtures build real dataclasses
(QualityReport) rather than dicts/mocks, so a test passing against these
fixtures is evidence the code handles the actual production type.
"""

from typing import Any, Dict

import pytest

from retriever.quality_gate import QualityReport, QualityStatus


@pytest.fixture
def quality_ok() -> QualityReport:
    return QualityReport(
        status=QualityStatus.QUALITY_OK,
        reason="Evidence sufficient for LLM reasoning",
        confidence_score=0.82,
        has_temporal_signal=False,
    )


@pytest.fixture
def quality_weak() -> QualityReport:
    return QualityReport(
        status=QualityStatus.QUALITY_WEAK,
        reason="Low semantic similarity for factual query",
        confidence_score=0.31,
        has_temporal_signal=False,
    )


@pytest.fixture
def quality_empty() -> QualityReport:
    return QualityReport(
        status=QualityStatus.QUALITY_EMPTY,
        reason="No evidence retrieved",
        confidence_score=0.0,
        has_temporal_signal=False,
    )


@pytest.fixture
def chunk():
    """Factory for evidence chunks matching the keys CapabilityAssessor reads."""

    def _make(
        entity: str,
        title: str | None = None,
        score: float = 0.8,
    ) -> Dict[str, Any]:
        return {
            "retrieval_context": entity,
            "source_title": title or entity,
            "score": score,
        }

    return _make
