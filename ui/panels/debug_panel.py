"""
Debug Panel — Inspection Mode Toggle (Sidebar Control)

Responsibility:
- Provide a single, global toggle to enable or disable debug / inspection mode.
- Act as a lightweight control surface for Tier 2 (Trace Panel) visibility.

Architectural Role (Phase 4):
- Dumb control component.
- Pure UI state binding with zero business logic.
- Completely decoupled from backend execution and results.

Contract Consumption:
- None.
- This panel does NOT read or inspect `ExecutionResult`.

State Interactions:
- Writes (via Streamlit binding):
  - `debug_enabled`

Constraints:
- Must render in the Streamlit sidebar.
- Must use direct state binding (`key="debug_enabled"`).
- Must not contain conditionals or callbacks.
"""

import streamlit as st


def render_debug_controls() -> None:
    """
    Render the debug mode toggle in the sidebar.

    This function relies entirely on Streamlit's native state
    handling and introduces no logic of its own.
    """

    with st.sidebar:
        st.checkbox(
            "Enable Debug Mode",
            key="debug_enabled",
        )
