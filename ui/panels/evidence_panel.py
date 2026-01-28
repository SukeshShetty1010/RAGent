"""
Evidence Panel — Source & Citation Viewer (Tier 1)

Responsibility:
- Render the exact evidence chunks used by the backend engine.
- Provide transparent, read-only inspection of supporting sources.
- Act as the system’s “Proof Layer” for user trust and verification.

Architectural Role (Phase 4):
- Tier 1 (Core User Experience) component.
- Strict *Dumb Component*:
  - Reads from Streamlit session state only.
  - Performs zero computation, ranking, filtering, or mutation.
  - Renders backend-provided data verbatim and in order.

Contract Consumption:
- Reads from `st.session_state.last_execution_result["evidence"]`:
  - Each item is an opaque evidence dictionary supplied by the backend.

Failure & Empty-State Semantics:
- If no execution result exists → render nothing.
- If execution exists but evidence list is empty → render a soft
  “No sources cited.” message.
- Must never crash the UI or raise exceptions.

Forbidden Actions:
- No re-ordering, deduplication, or scoring.
- No backend calls or state mutation.
- No truncation or semantic transformation of evidence text.
"""

from __future__ import annotations

import streamlit as st


def render_evidence_panel() -> None:
    """
    Render the evidence / sources panel.

    Safe to call on every Streamlit rerun.
    This function is strictly read-only and fail-soft.
    """

    result = st.session_state.get("last_execution_result")
    if result is None:
        return

    evidence_list = result.get("evidence", [])

    with st.container():
        st.markdown("### Sources")

        # --------------------------------------------------
        # Empty state (execution happened, no evidence)
        # --------------------------------------------------
        if not evidence_list:
            st.caption("No sources cited.")
            return

        # --------------------------------------------------
        # Evidence browsing (immutable backend order)
        # --------------------------------------------------
        for idx, evidence in enumerate(evidence_list, start=1):
            source_title = (
                evidence.get("source_title")
                or evidence.get("source")
                or "Unknown Source"
            )
            content = evidence.get("content", "")

            with st.expander(f"{idx}. {source_title}", expanded=False):
                if content:
                    st.markdown(content)
                else:
                    st.caption("No content available for this source.")
