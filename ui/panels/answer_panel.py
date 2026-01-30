# ============================================================
# ui/panels/answer_panel.py
# Primary Answer Output Surface (Tier 1)
# ============================================================

from __future__ import annotations

import streamlit as st


def render_answer_panel() -> None:
    """
    Render the primary answer panel.

    Architectural guarantees:
    - Read-only renderer
    - No inference or computation
    - Verbatim answer display
    - Safe on every Streamlit rerun
    """

    result = st.session_state.get("last_execution_result")
    if result is None:
        return

    final_answer = result.get("final_answer", "")

    kpis = result.get("kpis", {}) or {}
    quality_status = kpis.get("quality_status")
    answer_capability = kpis.get("answer_capability")

    with st.container():
        st.markdown("### Answer")

        # --------------------------------------------------
        # Capability cue (HONESTY FIRST)
        # --------------------------------------------------
        if answer_capability == "insufficient":
            st.error(
                "⚠️ The system does not have enough reliable information "
                "to answer this question safely."
            )

        elif answer_capability == "partial":
            st.warning(
                "⚠️ This answer is **partial**. Some information may be "
                "missing due to limited supporting evidence."
            )

        elif answer_capability == "full":
            st.success(
                "✅ This answer is fully supported by the available evidence."
            )

        # --------------------------------------------------
        # Evidence quality cue (SECONDARY)
        # --------------------------------------------------
        if quality_status == "QUALITY_WEAK":
            st.info(
                "Supporting evidence was limited or weak."
            )
        elif quality_status == "QUALITY_EMPTY":
            st.info(
                "No strong supporting evidence was found."
            )
        elif quality_status == "QUALITY_OK":
            st.info(
                "Answer generated based on retrieved evidence."
            )

        # --------------------------------------------------
        # Verbatim answer rendering (NO TRANSFORMATION)
        # --------------------------------------------------
        st.markdown(final_answer)
