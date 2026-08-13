"""
Lightweight Observability & Latency Profiling
--------------------------------------------

Design goals:
- Extremely low overhead (observer effect aware)
- Thread-safe
- Zero external dependencies
- Structured, machine-readable output

This module provides:
- ProfileBlock: nested wall-clock timing
- MetricsRegistry: counters, distributions, categorical metrics
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional
from contextlib import ContextDecorator


# ============================================================
# Shared constants
# ============================================================

# Separator used to join nested ProfileBlock names into a single
# latency:: metric key. Defined once and imported everywhere a
# ProfileBlock path is produced or parsed, so the two sides can never
# drift out of sync again.
PROFILE_PATH_SEP = " -> "


# ============================================================
# Metric Payloads
# ============================================================

@dataclass
class CounterMetric:
    name: str
    value: int = 0


@dataclass
class DistributionMetric:
    name: str
    values: List[float] = field(default_factory=list)


@dataclass
class CategoricalMetric:
    name: str
    values: Dict[str, int] = field(default_factory=dict)


# ============================================================
# Metrics Registry (Singleton)
# ============================================================

class MetricsRegistry:
    _instance: "MetricsRegistry | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._counters: Dict[str, CounterMetric] = {}
        self._distributions: Dict[str, DistributionMetric] = {}
        self._categoricals: Dict[str, CategoricalMetric] = {}
        self._registry_lock = threading.Lock()

    # --------------------------------------------------------
    # Singleton Access
    # --------------------------------------------------------
    @classmethod
    def get(cls) -> "MetricsRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------
    def inc(self, name: str, amount: int = 1) -> None:
        with self._registry_lock:
            metric = self._counters.setdefault(name, CounterMetric(name))
            metric.value += amount

    # --------------------------------------------------------
    # Distributions
    # --------------------------------------------------------
    def observe(self, name: str, value: float) -> None:
        with self._registry_lock:
            metric = self._distributions.setdefault(
                name, DistributionMetric(name)
            )
            metric.values.append(value)

    # --------------------------------------------------------
    # Categoricals
    # --------------------------------------------------------
    def record(self, name: str, label: str) -> None:
        with self._registry_lock:
            metric = self._categoricals.setdefault(
                name, CategoricalMetric(name)
            )
            metric.values[label] = metric.values.get(label, 0) + 1

    # --------------------------------------------------------
    # Point lookups
    # --------------------------------------------------------
    def last(self, name: str) -> float | None:
        """Most recently observed value for a distribution metric, or None."""
        with self._registry_lock:
            dist = self._distributions.get(name)
            if dist is None or not dist.values:
                return None
            return dist.values[-1]

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------
    def generate_report(self) -> Dict[str, Any]:
        with self._registry_lock:
            return {
                "counters": {
                    k: v.value for k, v in self._counters.items()
                },
                "distributions": {
                    k: {
                        "count": len(v.values),
                        "avg": (
                            sum(v.values) / len(v.values)
                            if v.values else 0.0
                        ),
                        "min": min(v.values) if v.values else None,
                        "max": max(v.values) if v.values else None,
                    }
                    for k, v in self._distributions.items()
                },
                "categoricals": {
                    k: v.values for k, v in self._categoricals.items()
                },
            }


# ============================================================
# Optional external span hooks (e.g. a tracing backend)
# ============================================================
#
# ProfileBlock stays free of any external dependency by construction —
# a caller (e.g. utils/tracing.py) can register a pair of callables here
# to mirror every ProfileBlock into an external span, without this
# module importing anything beyond the standard library.
_span_enter_hook: Optional[Callable[[str], Any]] = None
_span_exit_hook: Optional[Callable[[str, Any], None]] = None


def set_span_hooks(
    enter_hook: Optional[Callable[[str], Any]],
    exit_hook: Optional[Callable[[str, Any], None]],
) -> None:
    """Register enter/exit callables invoked around every ProfileBlock.

    enter_hook(name) -> token (opaque, passed back to exit_hook).
    exit_hook(name, token) -> None.
    Pass (None, None) to clear.
    """
    global _span_enter_hook, _span_exit_hook
    _span_enter_hook = enter_hook
    _span_exit_hook = exit_hook


# ============================================================
# ProfileBlock (Nested Timing)
# ============================================================

class ProfileBlock(ContextDecorator):
    """
    Ultra-lightweight profiling context manager.

    Example:
        with ProfileBlock("Orchestrator"):
            ...
    """

    _thread_local = threading.local()

    def __init__(self, name: str) -> None:
        self.name = name
        self.start: float | None = None
        self._span_token: Any = None

    def __enter__(self) -> "ProfileBlock":
        self.start = time.perf_counter()

        stack = getattr(self._thread_local, "stack", None)
        if stack is None:
            stack = []
            self._thread_local.stack = stack

        stack.append(self.name)

        if _span_enter_hook is not None:
            self._span_token = _span_enter_hook(self.name)

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end = time.perf_counter()
        duration_ms = (end - (self.start or end)) * 1000.0

        path = PROFILE_PATH_SEP.join(self._thread_local.stack)

        MetricsRegistry.get().observe(
            f"latency::{path}", duration_ms
        )

        self._thread_local.stack.pop()

        if _span_exit_hook is not None:
            _span_exit_hook(self.name, self._span_token)
