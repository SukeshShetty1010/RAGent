# ============================================================
# ui/panels/evidence_panel.py
# Evidence Panel — Trust-Focused Proof Layer
# ============================================================

from __future__ import annotations

import streamlit as st


def render_evidence_panel() -> None:
    """
    Render the evidence / sources panel.

    Design guarantees:
    - Read-only, fail-soft renderer
    - No reordering, filtering, or inference
    - Backend evidence shown verbatim
    - Collapsed, trust-first presentation
    """

    result = st.session_state.get("last_execution_result")
    if result is None:
        return

    evidence_list = result.get("evidence", []) or []

    with st.container():
        st.subheader("Evidence")

        # --------------------------------------------------
        # Empty state (execution occurred, no evidence)
        # --------------------------------------------------
        if not evidence_list:
            st.caption(
                "No supporting sources were cited for this response."
            )
            return

        st.caption(
            f"{len(evidence_list)} source(s) used to generate this answer."
        )

        # --------------------------------------------------
        # Evidence list (collapsed, per-source)
        # --------------------------------------------------
        for idx, evidence in enumerate(evidence_list, start=1):
            source_title = (
                evidence.get("source_title")
                or evidence.get("source")
                or "Unknown Source"
            )

            source_type = evidence.get("source_type", "local")
            content = evidence.get("content", "")

            label = f"{idx}. {source_title} ({source_type})"

            with st.expander(label, expanded=False):
                if content and content.strip():
                    st.markdown(content)
                else:
                    st.caption("No content available for this source.")
