"""
tests/test_capability_assessor.py

Hermetic tests for the honesty gate (CapabilityAssessor.assess()).

These tests pass a real QualityReport dataclass (see conftest.py fixtures),
not a dict — the bug this suite exists to catch is exactly that assess()
used to call quality.get(...) on an object with no .get() method.
"""

import pytest

from agent.capability.capability_assessor import CapabilityAssessor
from agent.capability.capability_types import AnswerCapability
from agent.intent.intent_signals import IntentSignal
from utils.observability import MetricsRegistry

pytestmark = pytest.mark.unit


@pytest.fixture
def assessor() -> CapabilityAssessor:
    return CapabilityAssessor()


def _errors() -> int:
    report = MetricsRegistry.get().generate_report()
    return report["counters"].get("capability_assess_errors", 0)


def test_no_evidence_is_insufficient(assessor, quality_ok):
    result = assessor.assess(
        intent_signals=set(),
        evidence=[],
        quality=quality_ok,
    )
    assert result == AnswerCapability.INSUFFICIENT


def test_quality_empty_is_insufficient(assessor, quality_empty, chunk):
    result = assessor.assess(
        intent_signals=set(),
        evidence=[chunk("Far Cry 5")],
        quality=quality_empty,
    )
    assert result == AnswerCapability.INSUFFICIENT


def test_quality_weak_is_partial(assessor, quality_weak, chunk):
    result = assessor.assess(
        intent_signals=set(),
        evidence=[chunk("Far Cry 5")],
        quality=quality_weak,
    )
    assert result == AnswerCapability.PARTIAL


def test_quality_ok_is_full(assessor, quality_ok, chunk):
    result = assessor.assess(
        intent_signals=set(),
        evidence=[chunk("Far Cry 5")],
        quality=quality_ok,
    )
    assert result == AnswerCapability.FULL


def test_comparison_single_entity_is_insufficient(assessor, quality_ok, chunk):
    result = assessor.assess(
        intent_signals={IntentSignal.COMPARISON},
        evidence=[chunk("Far Cry 5"), chunk("Far Cry 5")],
        quality=quality_ok,
    )
    assert result == AnswerCapability.INSUFFICIENT


def test_comparison_balanced_two_entities_is_full(assessor, quality_ok, chunk):
    evidence = [
        chunk("Far Cry 5"),
        chunk("Far Cry 5"),
        chunk("Assassin's Creed Valhalla"),
        chunk("Assassin's Creed Valhalla"),
    ]
    result = assessor.assess(
        intent_signals={IntentSignal.COMPARISON},
        evidence=evidence,
        quality=quality_ok,
    )
    assert result == AnswerCapability.FULL


def test_comparison_unbalanced_is_partial(assessor, quality_ok, chunk):
    evidence = (
        [chunk("Far Cry 5")] * 5
        + [chunk("Assassin's Creed Valhalla")] * 1
    )
    result = assessor.assess(
        intent_signals={IntentSignal.COMPARISON},
        evidence=evidence,
        quality=quality_ok,
    )
    assert result == AnswerCapability.PARTIAL


def test_listicle_with_quality_ok_is_full(assessor, quality_ok, chunk):
    result = assessor.assess(
        intent_signals={IntentSignal.LISTICLE},
        evidence=[chunk("Far Cry 5")],
        quality=quality_ok,
    )
    assert result == AnswerCapability.FULL


def test_listicle_with_quality_weak_is_partial(assessor, quality_weak, chunk):
    """Changed rule: a listicle no longer overrides a WEAK-evidence downgrade to FULL."""
    result = assessor.assess(
        intent_signals={IntentSignal.LISTICLE},
        evidence=[chunk("Far Cry 5")],
        quality=quality_weak,
    )
    assert result == AnswerCapability.PARTIAL


def test_temporal_without_signal_is_partial(assessor, quality_ok, chunk):
    result = assessor.assess(
        intent_signals={IntentSignal.TEMPORAL},
        evidence=[chunk("Far Cry 5")],
        quality=quality_ok,
    )
    assert result == AnswerCapability.PARTIAL


def test_no_silent_errors_across_all_cases(assessor, quality_ok, quality_weak, quality_empty, chunk):
    MetricsRegistry.get()._counters.clear()

    cases = [
        (set(), [], quality_ok),
        (set(), [chunk("Far Cry 5")], quality_empty),
        (set(), [chunk("Far Cry 5")], quality_weak),
        (set(), [chunk("Far Cry 5")], quality_ok),
        ({IntentSignal.COMPARISON}, [chunk("Far Cry 5")], quality_ok),
        ({IntentSignal.LISTICLE}, [chunk("Far Cry 5")], quality_weak),
        ({IntentSignal.TEMPORAL}, [chunk("Far Cry 5")], quality_ok),
    ]
    for signals, evidence, quality in cases:
        assessor.assess(intent_signals=signals, evidence=evidence, quality=quality)

    assert _errors() == 0
