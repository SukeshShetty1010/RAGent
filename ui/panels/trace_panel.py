# ============================================================
# ui/panels/trace_panel.py
# Trace Panel — Engineering Inspection Layer (FINAL)
# ============================================================

from __future__ import annotations

import streamlit as st


def render_trace_panel() -> None:
    """
    Render the internal execution trace for engineering inspection.

    Visibility rules:
    - Render ONLY if debug / inspection mode is enabled
    - Render ONLY if an execution result exists

    Guarantees:
    - Read-only
    - No state mutation
    - No backend calls
    - No computation or inference

    IMPORTANT:
    - This panel exposes INTERNAL diagnostics
    - Not intended for end-user interpretation
    """

    # --------------------------------------------------
    # Gate 1: Debug / inspection mode
    # --------------------------------------------------
    if not st.session_state.get("debug_enabled", False):
        return

    # --------------------------------------------------
    # Gate 2: Execution result must exist
    # --------------------------------------------------
    result = st.session_state.get("last_execution_result")
    if result is None:
        return

    agent_decisions = result.get("agent_decisions", {}) or {}
    raw_metrics = result.get("raw_metrics", {}) or {}

    with st.container():
        st.subheader("Internal System Trace")
        st.caption(
            "Engineering inspection view. "
            "Displays internal routing, quality signals, and raw observability. "
            "These signals are not user-facing guarantees."
        )

        # --------------------------------------------------
        # Routing & Retrieval Decisions
        # --------------------------------------------------
        with st.expander("Routing & Retrieval Decisions", expanded=False):
            st.json(
                {
                    "task": agent_decisions.get("task"),
                    "intent_signals": agent_decisions.get("intent_signals"),
                    "routing_reason": agent_decisions.get("routing_reason"),
                    "retrieval_strategy": agent_decisions.get(
                        "retrieval_strategy"
                    ),
                    "merge_state": agent_decisions.get("merge_state"),
                }
            )

        # --------------------------------------------------
        # Quality & Capability (INTERNAL DIAGNOSTICS)
        # --------------------------------------------------
        with st.expander(
            "Quality & Capability Diagnostics (Internal)",
            expanded=False,
        ):
            st.json(
                {
                    "quality_gate": agent_decisions.get("quality"),
                    "answer_capability": agent_decisions.get(
                        "answer_capability"
                    ),
                }
            )

        # --------------------------------------------------
        # Raw Observability Metrics (Deep Debug)
        # --------------------------------------------------
        with st.expander("Raw Observability Metrics", expanded=False):
            st.json(raw_metrics)
