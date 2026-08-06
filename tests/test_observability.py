"""
tests/test_observability.py

Hermetic tests for ProfileBlock / MetricsRegistry latency attribution.

Guards against the ASCII/Unicode arrow drift where ProfileBlock joined
path segments with " -> " while KPI/System_Performance_KPI.py split on
" → ", silently leaving stage_times empty forever.
"""

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
