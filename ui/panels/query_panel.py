# ui/panels/query_panel.py
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

import streamlit as st


def render_panel() -> None:
    """
    Render the query input panel.

    This function only mutates intent-related session state.
    Execution is handled by the application controller (`ui/app.py`).
    """

    query_value = st.text_input(
        "Enter your query",
        value=st.session_state.current_query or "",
        disabled=st.session_state.is_running,
    )

    run_clicked = st.button(
        "Run Analysis",
        disabled=st.session_state.is_running,
    )

    if run_clicked:
        st.session_state.current_query = query_value

        if query_value and query_value.strip():
            st.session_state.is_running = True
