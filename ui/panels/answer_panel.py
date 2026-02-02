# ============================================================
# ui/panels/answer_panel.py
# Answer Panel — Confidence-First Rendering (FINAL)
# ============================================================

from __future__ import annotations

import streamlit as st

from ui.utils.formatting import (
    format_answer_confidence,
    format_quality_status,
)


def render_answer_panel() -> None:
    """
    Render the primary answer surface.

    UI guarantees:
    - Confidence-first (not capability-first)
    - No error / warning semantics for PARTIAL
    - Honest degradation framed as a strength
    - Read-only, presentation-only
    """

    result = st.session_state.get("last_execution_result")
    if result is None:
        return

    final_answer = result.get("final_answer") or ""
    kpis = result.get("kpis", {}) or {}

    answer_capability = kpis.get("answer_capability")
    quality_status = kpis.get("quality_status")

    confidence = format_answer_confidence(answer_capability)

    with st.container():
        st.subheader("Answer")

        # ==================================================
        # 1. Answer Confidence (PRIMARY SIGNAL)
        # ==================================================
        st.markdown(
            f"### {confidence['label']}\n"
            f"{confidence['description']}"
        )

        # ==================================================
        # 2. Evidence Quality (SECONDARY CONTEXT)
        # ==================================================
        if quality_status:
            st.caption(
                f"Evidence quality: "
                f"{format_quality_status(quality_status)}"
            )

        st.divider()

        # ==================================================
        # 3. Verbatim Answer (NO TRANSFORMATION)
        # ==================================================
        if final_answer.strip():
            st.markdown(final_answer)
        else:
            st.caption("No answer text was produced.")
