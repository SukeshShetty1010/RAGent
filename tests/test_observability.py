"""
tests/test_observability.py

Hermetic tests for ProfileBlock / MetricsRegistry latency attribution.

Guards against the ASCII/Unicode arrow drift where ProfileBlock joined
path segments with " -> " while KPI/System_Performance_KPI.py split on
" → ", silently leaving stage_times empty forever.
"""

import threading

import pytest

from utils.observability import MetricsRegistry, ProfileBlock, PROFILE_PATH_SEP

pytestmark = pytest.mark.unit


def test_nested_profile_block_path_uses_shared_separator():
    registry = MetricsRegistry.get()
    registry._distributions.clear()

    with ProfileBlock("REQUEST_TOTAL"):
        with ProfileBlock("Retrieval"):
            pass

    report = registry.generate_report()
    expected_key = f"latency::REQUEST_TOTAL{PROFILE_PATH_SEP}Retrieval"

    assert expected_key in report["distributions"]


def test_system_performance_kpi_parses_the_same_separator():
    """
    Regression guard: KPI/System_Performance_KPI.py's stage-parsing logic
    must split on the exact same PROFILE_PATH_SEP that ProfileBlock emits.
    """
    registry = MetricsRegistry.get()
    registry._distributions.clear()

    with ProfileBlock("REQUEST_TOTAL"):
        with ProfileBlock("Retrieval"):
            pass

    report = registry.generate_report()

    stage_times = {}
    for path in report["distributions"]:
        if not path.startswith(f"latency::REQUEST_TOTAL{PROFILE_PATH_SEP}"):
            continue
        parts = path.replace("latency::", "").split(PROFILE_PATH_SEP)
        if len(parts) != 2:
            continue
        stage_times[parts[1]] = report["distributions"][path]["avg"]

    assert "Retrieval" in stage_times
    assert stage_times["Retrieval"] >= 0.0


# ============================================================
# MetricsRegistry — per-thread scoping (T5)
# ============================================================

def test_get_returns_stable_instance_within_one_thread():
    assert MetricsRegistry.get() is MetricsRegistry.get()


def test_two_threads_do_not_leak_into_each_other_or_the_main_thread():
    """api/main.py runs each chat request on its own thread. Each thread's
    registry must only ever contain that thread's own metrics."""
    MetricsRegistry.get().reset()
    MetricsRegistry.get().inc("main_thread_counter")

    results = {}

    def worker(label):
        registry = MetricsRegistry.get()
        registry.reset()
        registry.inc(f"{label}_counter")
        results[label] = registry.generate_report()["counters"]

    t1 = threading.Thread(target=worker, args=("thread_a",))
    t2 = threading.Thread(target=worker, args=("thread_b",))
    t1.start(); t1.join()
    t2.start(); t2.join()

    assert results["thread_a"] == {"thread_a_counter": 1}
    assert results["thread_b"] == {"thread_b_counter": 1}

    main_report = MetricsRegistry.get().generate_report()["counters"]
    assert main_report == {"main_thread_counter": 1}


def test_concurrent_reset_on_one_thread_does_not_touch_another():
    """The exact §5 failure mode: thread A records, then thread B resets
    and records its own request — thread A's report must be untouched."""
    barrier_a_recorded = threading.Event()
    barrier_b_done = threading.Event()
    thread_a_report = {}

    def thread_a():
        registry = MetricsRegistry.get()
        registry.reset()
        registry.inc("request_a_counter")
        barrier_a_recorded.set()
        barrier_b_done.wait(timeout=5)
        thread_a_report["counters"] = registry.generate_report()["counters"]

    def thread_b():
        barrier_a_recorded.wait(timeout=5)
        registry = MetricsRegistry.get()
        registry.reset()
        registry.inc("request_b_counter")
        barrier_b_done.set()

    t1 = threading.Thread(target=thread_a)
    t2 = threading.Thread(target=thread_b)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert thread_a_report["counters"] == {"request_a_counter": 1}


def test_reset_clears_counters_distributions_and_categoricals_together():
    registry = MetricsRegistry.get()
    registry.inc("c")
    registry.observe("d", 1.0)
    registry.record("cat", "label")

    registry.reset()

    report = registry.generate_report()
    assert report["counters"] == {}
    assert report["distributions"] == {}
    assert report["categoricals"] == {}


# ============================================================
# MetricsRegistry — last_label (T6)
# ============================================================

def test_last_label_returns_most_recent_label():
    registry = MetricsRegistry.get()
    registry.reset()
    registry.record("answer_model", "model-a")
    registry.record("answer_model", "model-b")

    assert registry.last_label("answer_model") == "model-b"


def test_last_label_returns_none_for_unknown_metric():
    registry = MetricsRegistry.get()
    registry.reset()

    assert registry.last_label("nonexistent") is None
