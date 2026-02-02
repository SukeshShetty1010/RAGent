# ui/panels/query_panel.py
"""
Query Panel — Command Bar Style

Responsibility:
- Collects user intent (query text).
- Signals *when* execution should start.
- UX-only polish: no backend logic, no result rendering.

Design Intent:
- Feel like a command bar, not a form.
- Guide users toward supported query types.
"""

from __future__ import annotations

import streamlit as st


# --------------------------------------------------
# Example Queries (Click-to-Fill)
# --------------------------------------------------
EXAMPLE_QUERIES = {
    "Comparison": "Compare the storyline of Far Cry 5 and Assassin’s Creed Valhalla",
    "Factual": "What is the release date of Far Cry 5?",
    "Listicle": "Top 5 things to do in Far Cry 5",
    "Temporal": "Latest update for Assassin’s Creed Valhalla",
}


def render_panel() -> None:
    """
    Render the query input panel.

    This panel:
    - Reads/writes intent-related session state only
    - Never executes backend logic
    """

    # --------------------------------------------------
    # Fail-safe state access
    # --------------------------------------------------
    current_query = st.session_state.get("current_query", "")
    is_running = st.session_state.get("is_running", False)

    # --------------------------------------------------
    # Command Bar Input
    # --------------------------------------------------
    st.markdown("### Ask a question")

    query_value = st.text_input(
        label="",
        value=current_query or "",
        disabled=is_running,
        placeholder="Type a comparison, factual question, list, or update query…",
    )

    # --------------------------------------------------
    # Supported Query Types (Guidance)
    # --------------------------------------------------
    st.caption(
        "Supported query types: "
        "**Comparison · Factual · Listicle · Temporal**"
    )

    # --------------------------------------------------
    # Example Queries (Click-to-Fill)
    # --------------------------------------------------
    with st.container():
        cols = st.columns(len(EXAMPLE_QUERIES))
        for col, (label, example) in zip(cols, EXAMPLE_QUERIES.items()):
            if col.button(label, disabled=is_running):
                st.session_state.current_query = example
                st.session_state.is_running = True
                return

    # --------------------------------------------------
    # Run Trigger
    # --------------------------------------------------
    run_clicked = st.button(
        "Run ▶",
        disabled=is_running,
        type="primary",
    )

    # --------------------------------------------------
    # Intent Signaling (NO execution here)
    # --------------------------------------------------
    if run_clicked:
        st.session_state.current_query = query_value

        if query_value and query_value.strip():
            st.session_state.is_running = True
