# ============================================================
# ui/panels/debug_panel.py
# Debug Panel — Engineering Inspection Controls (FINAL)
# ============================================================

import streamlit as st


def render_debug_controls() -> None:
    """
    Render global debug / inspection controls.

    Design intent:
    - Explicit separation between user-facing UI and engineering diagnostics
    - Single, opt-in control surface for inspection
    - Zero business logic

    IMPORTANT:
    - These controls expose INTERNAL system diagnostics
    - Not intended for end-user interpretation
    """

    with st.sidebar:
        st.markdown("### Inspection Mode")

        st.checkbox(
            "Show internal system diagnostics",
            key="debug_enabled",
        )

        st.caption(
            "Engineering-only view. Enables internal routing decisions, "
            "quality gates, capability diagnostics, and raw observability "
            "metrics. These signals are not user-facing guarantees."
        )
