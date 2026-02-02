# ============================================================
# tests/KPI/Unified_KPI_Runner.py
# EXECUTIVE RAG HEALTH DASHBOARD (MASTER RUNNER)
# ============================================================

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# KPI Imports
# ------------------------------------------------------------

from KPI.Context_Engineering_KPI import ContextEngineeringKPI
from KPI.Faith_Fair_KPI import FaithFairKPI
from KPI.Retrieval_Quality_KPI import RetrievalQualityKPI
from KPI.System_Performance_KPI import SystemPerformanceKPI
from KPI.Intent_Agent_Control import ResumeKPIDashboard

# ------------------------------------------------------------
# ANSI formatting (high-contrast executive view)
# ------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"


# ============================================================
# Unified Dashboard Orchestrator
# ============================================================

class UnifiedDashboard:
    """
    Master orchestration layer for all resume-grade KPIs.

    Executive ordering principle:
    1. Context efficiency & cost control
    2. Honesty, faithfulness & safety
    3. Retrieval effectiveness
    4. System performance & latency
    5. Intent routing & determinism
    """

    def run(self) -> None:
        # Silence noisy logs globally
        logging.getLogger().setLevel(logging.WARNING)

        print(
            f"\n{BOLD}{CYAN}"
            f"=== RAG SYSTEM EXECUTIVE DASHBOARD ==="
            f"{RESET}\n"
        )

        # =====================================================
        # 1. Context Engineering (COST + RELIABILITY FIRST)
        # =====================================================
        print(f"{BOLD}{'=' * 70}{RESET}")
        ContextEngineeringKPI().run()

        # =====================================================
        # 2. Faithfulness, Honesty & Safety
        # =====================================================
        print(f"{BOLD}{'=' * 70}{RESET}")
        try:
            FaithFairKPI().run()
        except Exception as e:
            print(
                f"{RED}Faith/Fair KPI failed safely: "
                f"{type(e).__name__}: {e}{RESET}\n"
            )

        # =====================================================
        # 3. Retrieval Quality
        # =====================================================
        print(f"{BOLD}{'=' * 70}{RESET}")
        try:
            RetrievalQualityKPI().run()
        except Exception as e:
            print(
                f"{RED}Retrieval Quality KPI failed safely: "
                f"{type(e).__name__}: {e}{RESET}\n"
            )

        # =====================================================
        # 4. System Performance & Latency
        # =====================================================
        print(f"{BOLD}{'=' * 70}{RESET}")
        try:
            SystemPerformanceKPI().run()
        except Exception as e:
            print(
                f"{RED}System Performance KPI failed safely: "
                f"{type(e).__name__}: {e}{RESET}\n"
            )

        # =====================================================
        # 5. Intent Routing, Determinism & Stability
        # =====================================================
        print(f"{BOLD}{'=' * 70}{RESET}")
        intent_dashboard = ResumeKPIDashboard()
        metrics = intent_dashboard.run()
        ResumeKPIDashboard.print_dashboard(metrics)

        print(f"{BOLD}{'=' * 70}{RESET}\n")


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    UnifiedDashboard().run()
