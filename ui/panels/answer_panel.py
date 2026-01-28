"""
Answer Panel — Primary Output Surface (Tier 1)

Responsibility:
- Display the final answer text produced by the backend engine.
- Provide immediate visual context based on the backend quality assessment.
- Act as the main, authoritative output surface for the user.

Architectural Role (Phase 4):
- Tier 1 (Core User Experience) component.
- Strictly a *Dumb / Read-Only* renderer:
  - Consumes backend state.
  - Performs zero computation or inference.
  - Never mutates state or influences execution flow.

Contract Consumption:
- Reads from `st.session_state.last_execution_result`:
  - `final_answer`
  - `kpis.quality_status`

Failure Semantics:
- If no execution result exists → render nothing.
- The answer is always rendered verbatim, even when quality is low.

Quality → Visual Mapping:
- QUALITY_OK    → Neutral / informational display
- QUALITY_WEAK  → Warning alert
- QUALITY_EMPTY → Error alert

Forbidden Actions:
- Must never transform, truncate, or summarize `final_answer`.
- Must never compute success/failure locally.
- Must never access `raw_metrics`, `agent_decisions`, or trace data.
"""

from __future__ import annotations

import streamlit as st


def render_answer_panel() -> None:
    """
    Render the primary answer panel.

    Safe to call on every Streamlit rerun.
    This function is fail-soft and strictly read-only.
    """

    result = st.session_state.get("last_execution_result")
    if result is None:
        return

    final_answer = result.get("final_answer", "")
    quality_status = (
        result.get("kpis", {}) or {}
    ).get("quality_status")

    with st.container():
        st.markdown("### Answer")

        # --------------------------------------------------
        # Quality cue (must appear before answer text)
        # --------------------------------------------------
        if quality_status == "QUALITY_WEAK":
            st.warning(
                "The answer was generated with limited supporting evidence."
            )
        elif quality_status == "QUALITY_EMPTY":
            st.error(
                "Insufficient evidence was found to answer this query reliably."
            )
        elif quality_status == "QUALITY_OK":
            st.info("Answer generated based on available evidence.")

        # --------------------------------------------------
        # Verbatim answer rendering (no transformation)
        # --------------------------------------------------
        st.markdown(final_answer)
