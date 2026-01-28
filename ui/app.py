"""
Phase 4 UI Orchestrator — RAGent
================================

Role:
- Declarative layout & composition layer for the Streamlit UI.
- Enforces the Smart-Container / Dumb-Component contract.
- Owns execution flow but delegates *all rendering* to panel components.

Phase 4 Guarantees:
- Clear visual hierarchy (Input → Results → Inspection).
- Unidirectional data flow: Engine → Session State → Panels.
- Fail-soft rendering when no execution result exists.
- Zero business logic or metric inspection in this file.

This file is the ONLY UI surface allowed to:
- Call RageEngine
- Transition `is_running`
- Write `last_execution_result`
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

    Ensures:
    - One engine per Streamlit session
    - No repeated initialization on reruns
    """
    return RageEngine()


# ---------------------------------------------------------
# Application Entrypoint
# ---------------------------------------------------------
def run_app() -> None:
    """
    Main UI entrypoint (Phase 4).

    Execution Order:
    1. Initialize session state
    2. Render global layout
    3. Observe intent (`is_running`)
    4. Execute engine if needed
    5. Render result & inspection panels
    """

    # -----------------------------------------------------
    # 1. Initialize canonical UI state (MUST be first)
    # -----------------------------------------------------
    initialize_state()

    # -----------------------------------------------------
    # 2. Sidebar: Debug Controls
    # -----------------------------------------------------
    debug_panel.render_debug_controls()

    # -----------------------------------------------------
    # 3. Global Header
    # -----------------------------------------------------
    st.title("RAGent")

    # -----------------------------------------------------
    # 4. Input Layer
    # -----------------------------------------------------
    query_panel.render_panel()

    # -----------------------------------------------------
    # 5. Execution Controller (Fail-Soft)
    # -----------------------------------------------------
    if st.session_state.is_running:
        with st.status("Running analysis…", expanded=False):
            engine = get_engine()

            try:
                result = engine.run(st.session_state.current_query)
                st.session_state.last_execution_result = result
            finally:
                st.session_state.is_running = False

    # -----------------------------------------------------
    # 6. Results Layer (Stable even if result is None)
    # -----------------------------------------------------
    kpi_panel.render_kpi_panel()
    answer_panel.render_answer_panel()
    evidence_panel.render_evidence_panel()

    # -----------------------------------------------------
    # 7. Inspection Layer (Debug / Trace)
    # -----------------------------------------------------
    trace_panel.render_trace_panel()


# ---------------------------------------------------------
# Script Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    run_app()
