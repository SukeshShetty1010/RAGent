import pytest

from agent.task_router import RouterDecision, TaskType
from agent.intent.intent_signals import IntentSignal
from retriever.strategy_selector import StrategySelector


@pytest.fixture
def selector():
    return StrategySelector()


@pytest.mark.unit
@pytest.mark.parametrize(
    "task,intent_signals",
    [
        (TaskType.COMPARISON, {IntentSignal.COMPARISON}),
        (TaskType.LISTICLE, {IntentSignal.LISTICLE}),
        (TaskType.FACTUAL, {IntentSignal.FACTUAL}),
        (TaskType.FACTUAL, {IntentSignal.FACTUAL, IntentSignal.COMPARISON}),
    ],
)
def test_allow_web_fallback_false_without_temporal_signal(selector, task, intent_signals):
    decision = RouterDecision(task=task, intent_signals=intent_signals, reason="test")
    config = selector.select(decision)
    assert config.allow_web_fallback is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "task,intent_signals",
    [
        (TaskType.COMPARISON, {IntentSignal.COMPARISON, IntentSignal.TEMPORAL}),
        (TaskType.LISTICLE, {IntentSignal.LISTICLE, IntentSignal.TEMPORAL}),
        (TaskType.FACTUAL, {IntentSignal.FACTUAL, IntentSignal.TEMPORAL}),
        (
            TaskType.FACTUAL,
            {IntentSignal.FACTUAL, IntentSignal.COMPARISON, IntentSignal.TEMPORAL},
        ),
    ],
)
def test_allow_web_fallback_true_with_temporal_signal(selector, task, intent_signals):
    decision = RouterDecision(task=task, intent_signals=intent_signals, reason="test")
    config = selector.select(decision)
    assert config.allow_web_fallback is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "intent_signals",
    [set(), {IntentSignal.TEMPORAL}],
)
def test_open_task_always_allows_web_fallback(selector, intent_signals):
    decision = RouterDecision(task=TaskType.OPEN, intent_signals=intent_signals, reason="test")
    config = selector.select(decision)
    assert config.allow_web_fallback is True


@pytest.mark.unit
def test_factual_mixed_intent_with_temporal_signal_regression(selector):
    """Regression test for AUDIT_TASKS.md T11: a FACTUAL query with a
    TEMPORAL signal (e.g. "latest patch notes for X") must be able to
    reach the web fallback path, not just OPEN-task queries."""
    decision = RouterDecision(
        task=TaskType.FACTUAL,
        intent_signals={IntentSignal.FACTUAL, IntentSignal.TEMPORAL},
        reason="Selected factual as primary task based on intent signals: factual, temporal",
    )
    config = selector.select(decision)
    assert config.allow_web_fallback is True
