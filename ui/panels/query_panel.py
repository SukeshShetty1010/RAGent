"""
Query Panel — Request Initiator

Responsibility:
- Collects user intent (the query text).
- Signals *when* an execution should start.
- Does NOT execute backend logic.
- Does NOT render results.

Architectural Role:
- Dumb component in a Smart-Container / Dumb-Component model.
- Purely state-driven and intent-focused.

Contract Consumption:
- None.
- This panel does NOT read `ExecutionResult` or any backend data.

State Interactions:
- Reads:
  - `current_query`
  - `is_running`
- Writes:
  - `current_query`
  - `is_running`

Interaction Flow:
1. User types a query into the input field.
2. User clicks "Run Analysis".
3. If the query is non-empty and no execution is in progress:
   - Update `current_query`.
   - Set `is_running = True`.

Forbidden Actions:
- Must never import or call `RageEngine`.
- Must never import backend schemas or results.
- Must never modify `last_execution_result`.
- Must never render answers, evidence, KPIs, or traces.
"""

from __future__ import annotations

import streamlit as st


def render_panel() -> None:
    """
    Render the query input panel.

    This function only mutates intent-related session state.
    Execution is handled by the application controller (`ui/app.py`).
    """

    # --------------------------------------------------
    # Fail-safe state access (critical for first render)
    # --------------------------------------------------
    current_query = st.session_state.get("current_query", "")
    is_running = st.session_state.get("is_running", False)

    # --------------------------------------------------
    # Input field (always renders)
    # --------------------------------------------------
    query_value = st.text_input(
        "Enter your query",
        value=current_query or "",
        disabled=is_running,
        placeholder="Ask a question to begin…",
    )

    # --------------------------------------------------
    # Run trigger
    # --------------------------------------------------
    run_clicked = st.button(
        "Run Analysis",
        disabled=is_running,
    )

    # --------------------------------------------------
    # Intent signaling (no execution here)
    # --------------------------------------------------
    if run_clicked:
        st.session_state.current_query = query_value

        if query_value and query_value.strip():
            st.session_state.is_running = True
