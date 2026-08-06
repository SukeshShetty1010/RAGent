#!/usr/bin/env python3
"""
PHASE 0 — UI READINESS VERIFICATION
==================================

This script is a HARD GATE for RageEngine.

It enforces:
1. Silence Protocol (no stdout pollution)
2. Schema Contract (stable, JSON-safe result)
3. Statelessness (no metric leakage between runs)

If ANY check fails → ENGINE IS NOT UI SAFE.
"""

from __future__ import annotations

import sys
import io
import traceback
from typing import Dict, Any
from contextlib import redirect_stdout

from colorama import Fore, Style, init

# ---------------------------------------------------------------------
# Colorama init
# ---------------------------------------------------------------------
init(autoreset=True)

# ---------------------------------------------------------------------
# Engine import (DO NOT MODIFY ENGINE)
# ---------------------------------------------------------------------
try:
    from engine.execution_engine import RageEngine
except Exception as exc:
    print(Fore.RED + "❌ FAILED TO IMPORT RageEngine")
    print(exc)
    sys.exit(1)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def fail(reason: str) -> None:
    print(Fore.RED + "❌ ENGINE UNSAFE")
    print(Fore.RED + f"Reason: {reason}")
    sys.exit(1)


def pass_check(msg: str) -> None:
    print(Fore.GREEN + "✔ " + msg)


# ---------------------------------------------------------------------
# TEST 1 — SILENCE PROTOCOL
# ---------------------------------------------------------------------
def test_silence(engine: RageEngine) -> Dict[str, Any]:
    """
    Capture ALL stdout during engine execution.
    ANY output = FAILURE.
    """
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        result = engine.run("What is the release date of Far Cry 5?")

    leaked_output = buffer.getvalue()

    if leaked_output.strip():
        fail(
            "Silence Protocol violated — engine printed to stdout:\n"
            + leaked_output
        )

    pass_check("Silence Protocol enforced (no stdout leakage)")
    return result


# ---------------------------------------------------------------------
# TEST 2 — SCHEMA CONTRACT
# ---------------------------------------------------------------------
def test_schema(result: Dict[str, Any]) -> None:
    required_keys = {
        "final_answer",
        "agent_decisions",
        "evidence",
        "kpis",
        "raw_metrics",
    }

    if not isinstance(result, dict):
        fail("Engine result is not a dictionary")

    missing = required_keys - result.keys()
    extra = result.keys() - required_keys

    if missing:
        fail(f"Missing required keys: {sorted(missing)}")

    if extra:
        fail(f"Unexpected extra keys present: {sorted(extra)}")

    if not isinstance(result["final_answer"], str):
        fail("final_answer must be a string")

    pass_check("Schema contract valid and stable")


# ---------------------------------------------------------------------
# TEST 3 — STATELESSNESS / METRIC ISOLATION
# ---------------------------------------------------------------------
def test_statelessness(engine: RageEngine) -> None:
    """
    Run engine twice.
    Metrics MUST reset between runs.
    """
    r1 = engine.run("What is the release date of Far Cry 5?")
    r2 = engine.run("What is the release date of Far Cry 5?")

    m2 = r2.get("raw_metrics")

    if not isinstance(m2, dict):
        fail("raw_metrics is not a dictionary")

    counters = m2.get("counters", {})

    if not isinstance(counters, dict):
        fail("raw_metrics.counters is not a dictionary")

    for name, value in counters.items():
        if not isinstance(value, int):
            fail(f"Metric '{name}' is not an int")

        if value > 1:
            fail(
                f"Metric leakage detected — counter '{name}' = {value} "
                "(expected ≤ 1 per request)"
            )

    pass_check("Statelessness confirmed (metrics reset per run)")


# ---------------------------------------------------------------------
# MAIN — GATEKEEPER
# ---------------------------------------------------------------------
def main() -> None:
    print(Style.BRIGHT + "\n=== PHASE 0: UI SANITY CHECK ===\n")

    try:
        engine = RageEngine()
    except Exception as exc:
        fail(f"Engine initialization failed: {exc}")

    try:
        # TEST 1
        result = test_silence(engine)

        # TEST 2
        test_schema(result)

        # TEST 3
        test_statelessness(engine)

    except SystemExit:
        raise
    except Exception:
        print(Fore.RED + "❌ UNHANDLED EXCEPTION DURING VERIFICATION")
        traceback.print_exc()
        sys.exit(1)

    print(
        Style.BRIGHT
        + Fore.GREEN
        + "\n✅ ENGINE READY FOR UI\n"
    )
    sys.exit(0)


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
