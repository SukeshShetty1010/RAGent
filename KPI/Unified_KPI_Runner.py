# ============================================================
# tests/KPI/Unified_KPI_Runner.py
# EXECUTIVE RAG HEALTH DASHBOARD (MASTER RUNNER)
# ============================================================

from __future__ import annotations

import logging

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
RESET = "\033[0m"


# ============================================================
# Unified Dashboard Orchestrator
# ============================================================

class UnifiedDashboard:
    """
    Master orchestration layer for all resume-grade KPIs.

    Runs:
    - Context Engineering KPIs
    - Faithfulness, Honesty & Safety KPIs
    - Retrieval Quality KPIs
    - System Performance KPIs
    - Intent, Routing & Stability KPIs

    Output:
    - Single continuous executive dashboard stream
    """

    def run(self) -> None:
        logging.getLogger().setLevel(logging.WARNING)

        print(
            f"\n{BOLD}{CYAN}"
            f"=== RAG SYSTEM EXECUTIVE DASHBOARD ==="
            f"{RESET}\n"
        )

        # =====================================================
        # Context Engineering
        # =====================================================
        print(f"{BOLD}{'=' * 70}{RESET}")
        ContextEngineeringKPI().run()

        # =====================================================
        # Faithfulness, Honesty & Safety
        # =====================================================
        print(f"{BOLD}{'=' * 70}{RESET}")
        FaithFairKPI().run()

        # =====================================================
        # Retrieval Quality
        # =====================================================
        print(f"{BOLD}{'=' * 70}{RESET}")
        RetrievalQualityKPI(k=5).run()

        # =====================================================
        # System Performance
        # =====================================================
        print(f"{BOLD}{'=' * 70}{RESET}")
        SystemPerformanceKPI().run()

        # =====================================================
        # Intent, Routing & Stability (Special Handling)
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
