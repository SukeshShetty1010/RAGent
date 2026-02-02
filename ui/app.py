# ============================================================
# ui/app.py
# Phase 4 UI Orchestrator — RAGent (EXECUTIVE-FIRST LAYOUT)
# ============================================================
"""
Role:
- Declarative layout & composition layer for the Streamlit UI.
- Enforces a clear visual hierarchy:
    1. Query
    2. Executive Summary (Trust Signals)
    3. Details (collapsed)

STRICT GUARANTEES:
- No business logic
- No metric inspection
- No backend coupling beyond engine.run()
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------
# State & Engine
# ---------------------------------------------------------
from ui.state import initialize_state
from engine.execution_engine import RageEngine

# ---------------------------------------------------------
# Panels (Dumb Components)
# ---------------------------------------------------------
from ui.panels import (
    query_panel,
    kpi_panel,
    answer_panel,
    evidence_panel,
    trace_panel,
    debug_panel,
)

# ---------------------------------------------------------
# Engine Singleton (DO NOT MODIFY)
# ---------------------------------------------------------
@st.cache_resource
def get_engine() -> RageEngine:
    """
    Cached RageEngine singleton.

    Guarantees:
    - One engine per Streamlit session
    - No repeated initialization on reruns
    """
    return RageEngine()


# ---------------------------------------------------------
# Application Entrypoint
# ---------------------------------------------------------
def run_app() -> None:
    """
    Main UI entrypoint.

    Visual Hierarchy:
    ┌──────────────────────────────┐
    │ 1. Query                     │
    ├──────────────────────────────┤
    │ 2. Executive Summary         │
    │    - Trust & Safety KPIs     │
    │    - Answer Confidence       │
    ├──────────────────────────────┤
    │ 3. Details (Collapsed)       │
    │    - Evidence                │
    │    - Internal Trace (Debug)  │
    └──────────────────────────────┘
    """

    # -----------------------------------------------------
    # 1. Initialize canonical UI state (MUST be first)
    # -----------------------------------------------------
    initialize_state()

    # -----------------------------------------------------
    # 2. Sidebar: Debug / Inspection Controls
    # -----------------------------------------------------
    debug_panel.render_debug_controls()

    # -----------------------------------------------------
    # 3. Global Header (Trust Anchor)
    # -----------------------------------------------------
    st.title("RAGent")
    st.caption(
        "Evidence-Bound · Deterministic · Honest Degradation · Measurable"
    )

    # =====================================================
    # ZONE 1 — QUERY (Primary Interaction)
    # =====================================================
    with st.container():
        query_panel.render_panel()

    # -----------------------------------------------------
    # Execution Controller (Fail-Soft)
    # -----------------------------------------------------
    if st.session_state.is_running:
        with st.status("Running analysis…", expanded=False):
            engine = get_engine()
            try:
                result = engine.run(st.session_state.current_query)
                st.session_state.last_execution_result = result
            finally:
                st.session_state.is_running = False

    # =====================================================
    # ZONE 2 — EXECUTIVE SUMMARY (ALWAYS VISIBLE)
    # =====================================================
    with st.container():
        st.subheader("Executive Summary")

        # Trust & Safety KPIs (Tier 1)
        kpi_panel.render_kpi_panel()

        # Answer rendered with confidence framing
        answer_panel.render_answer_panel()

    # =====================================================
    # ZONE 3 — DETAILS (COLLAPSED BY DEFAULT)
    # =====================================================
    with st.expander("Show details & system evidence", expanded=False):

        # Evidence (trust-building, non-essential upfront)
        evidence_panel.render_evidence_panel()

        # Internal diagnostics (engineering only)
        trace_panel.render_trace_panel()


# ---------------------------------------------------------
# Script Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    run_app()
