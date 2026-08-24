#!/usr/bin/env python3
"""
evaluation/run_kpi_suite.py — Instrumented, persisted re-run of KPI/Unified_KPI_Runner.py

T30 (AUDIT_TASKS.md): the 5-module KPI suite has never been re-run since the
Gemini embedding migration, reranker-provider swap, Gemini/Groq model swaps,
and the T7 engine reconciliation. It also has zero file/JSON output anywhere
in KPI/ — every module just print()s ANSI tables. This script imports the
same 5 classes KPI/Unified_KPI_Runner.py uses, in the same order, with the
same fault-tolerance semantics, but captures each module's stdout + timing +
exception state into a single dated JSON artifact so "does it still run" is
answerable without re-running.

This does not replace `python -m KPI.Unified_KPI_Runner` — that command still
means exactly what it always meant. This is an additive sibling entrypoint.
No files under KPI/ are imported for their side effects beyond `.run()`, and
none are edited.

Usage:
    python -m evaluation.run_kpi_suite
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).resolve().parent / "results"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _run_module(
    module_name: str,
    caught_by_real_runner: bool,
    call,
) -> Dict[str, Any]:
    """
    Execute one KPI module's callable with stdout captured, timed, and
    exception-isolated. `call` takes no args and returns the module's
    structured result (or None if it only prints).
    """
    buf = io.StringIO()
    structured: Optional[Any] = None
    status = "completed"
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    tb: Optional[str] = None

    start = time.monotonic()
    try:
        with contextlib.redirect_stdout(buf):
            structured = call()
    except Exception as e:  # noqa: BLE001 — deliberately broad, this is a diagnostic harness
        status = "crashed"
        exception_type = type(e).__name__
        exception_message = str(e)
        tb = traceback.format_exc()
    duration_s = round(time.monotonic() - start, 3)

    return {
        "module": module_name,
        "status": status,
        "caught_by_real_runner": caught_by_real_runner,
        "duration_s": duration_s,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "traceback": tb,
        "stdout": _strip_ansi(buf.getvalue()),
        "structured": structured,
    }


def _derive_overall_status(modules: List[Dict[str, Any]]) -> str:
    """A hard crash on an uncaught module means the real entrypoint
    (python -m KPI.Unified_KPI_Runner) would have died at that point and
    never reached later modules."""
    hard_crash = False
    any_failure = False
    for m in modules:
        if m["status"] == "crashed":
            any_failure = True
            if not m["caught_by_real_runner"]:
                hard_crash = True
                break  # modules after this point would never have run for real

    if hard_crash:
        return "hard_crash_before_completion"
    if any_failure:
        return "partial_failure"
    return "all_completed"


def main() -> None:
    from KPI.Context_Engineering_KPI import ContextEngineeringKPI
    from KPI.Faith_Fair_KPI import FaithFairKPI
    from KPI.Retrieval_Quality_KPI import RetrievalQualityKPI
    from KPI.System_Performance_KPI import SystemPerformanceKPI
    from KPI.Intent_Agent_Control import ResumeKPIDashboard
    from llm.gemini_client import GEMINI_MODEL
    from llm.ragent_client import _GROQ_MODEL as GROQ_MODEL

    reranker_provider = os.environ.get("RERANKER_PROVIDER", "local")

    modules: List[Dict[str, Any]] = []

    # 1. Context Engineering — uncaught in the real runner (Unified_KPI_Runner.py:70)
    modules.append(
        _run_module(
            "context_engineering",
            caught_by_real_runner=False,
            call=lambda: ContextEngineeringKPI().run(),
        )
    )

    # 2. Faithfulness / Honesty / Safety — caught in the real runner (line 77)
    modules.append(
        _run_module(
            "faith_fair",
            caught_by_real_runner=True,
            call=lambda: FaithFairKPI().run(),
        )
    )

    # 3. Retrieval Quality — caught in the real runner (line 89)
    modules.append(
        _run_module(
            "retrieval_quality",
            caught_by_real_runner=True,
            call=lambda: RetrievalQualityKPI().run(),
        )
    )

    # 4. System Performance & Latency — caught in the real runner (line 101)
    modules.append(
        _run_module(
            "system_performance",
            caught_by_real_runner=True,
            call=lambda: SystemPerformanceKPI().run(),
        )
    )

    # 5. Intent Routing, Determinism & Stability — uncaught in the real runner
    # (lines 112-114); the one module with structured output pre-rendering.
    def _run_intent_routing():
        dashboard = ResumeKPIDashboard()
        metrics = dashboard.run()
        ResumeKPIDashboard.print_dashboard(metrics)
        return [asdict(m) for m in metrics]

    modules.append(
        _run_module(
            "intent_routing",
            caught_by_real_runner=False,
            call=_run_intent_routing,
        )
    )

    total_duration_s = round(sum(m["duration_s"] for m in modules), 3)
    overall_status = _derive_overall_status(modules)

    artifact = {
        "date": date.today().isoformat(),
        "reranker_provider": reranker_provider,
        "gemini_model": GEMINI_MODEL,
        "groq_model": GROQ_MODEL,
        "embedding_model": "gemini-embedding-001",
        "total_duration_s": total_duration_s,
        "overall_status": overall_status,
        "modules": modules,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"kpi_suite_{reranker_provider}_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Ran KPI suite ({reranker_provider}) -> {out_path}")
    print(
        json.dumps(
            {
                "overall_status": overall_status,
                "total_duration_s": total_duration_s,
                "modules": [
                    {
                        "module": m["module"],
                        "status": m["status"],
                        "caught_by_real_runner": m["caught_by_real_runner"],
                        "duration_s": m["duration_s"],
                        "exception_type": m["exception_type"],
                    }
                    for m in modules
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
