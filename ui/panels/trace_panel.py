"""
Trace Panel — Execution Trace Inspector (Tier 2 / Debug Layer)

Responsibility:
- Provide deep, read-only visibility into the internal execution state
  of the RAGent pipeline.
- Expose backend reasoning artifacts strictly for inspection and debugging.
- Serve as the system’s “Inspection Layer” without affecting UX flow.

Architectural Role (Phase 4):
- Tier 2 (Debug / Inspector) component.
- Opt-in only: rendered exclusively when debug mode is enabled.
- Strictly passive:
  - No state mutation
  - No backend calls
  - No execution control

Contract Consumption:
- Reads from `st.session_state.last_execution_result`:
  - `agent_decisions`
  - `raw_metrics`

State Interactions:
- Reads only:
  - `debug_enabled`
  - `last_execution_result`

Visibility Rules:
- If `debug_enabled` is False → render nothing.
- If no execution result exists → render nothing.

Forbidden Actions:
- Must never modify `st.session_state`.
- Must never compute, infer, or transform backend data.
- Must never influence execution flow or UI state.
"""

from __future__ import annotations

import streamlit as st


def render_trace_panel() -> None:
    """
    Render the execution trace inspection panel.

    This function is fail-soft, read-only, and safe to call
    on every Streamlit rerun.
    """

    # --------------------------------------------------
    # Gate 1: Debug mode must be explicitly enabled
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
        st.markdown("### Execution Trace")

        # --------------------------------------------------
        # Routing & Strategy
        # --------------------------------------------------
        with st.expander("Routing & Strategy", expanded=False):
            st.json(
                {
                    "task": agent_decisions.get("task"),
                    "routing_reason": agent_decisions.get("routing_reason"),
                    "retrieval_strategy": agent_decisions.get(
                        "retrieval_strategy"
                    ),
                    "merge_state": agent_decisions.get("merge_state"),
                }
            )

        # --------------------------------------------------
        # Quality Assessment
        # --------------------------------------------------
        with st.expander("Quality Assessment", expanded=False):
            st.json(agent_decisions.get("quality", {}))

        # --------------------------------------------------
        # Raw Metrics (Deep Inspection)
        # --------------------------------------------------
        with st.expander("Raw Metrics", expanded=False):
            st.json(raw_metrics)
