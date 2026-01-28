"""
KPI Panel — Execution Scoreboard (Tier 1)

Responsibility:
- Render a compact, horizontal KPI ribbon immediately after execution.
- Provide transparent, high-level system feedback to the user.

Architectural Role (Phase 4):
- Tier 1 (Core User Experience) component.
- Strictly a *Dumb Component*:
  - Reads from Streamlit session state.
  - Performs zero computation, inference, or side effects.
  - Renders backend-provided values verbatim.

Contract Consumption:
- Reads from `st.session_state.last_execution_result["kpis"]`:
  - engine_latency_ms
  - quality_status
  - confidence_score
  - task_success

Failure Semantics:
- If no execution result exists, render nothing.
- If individual fields are missing, render stable "N/A" values.
"""

from __future__ import annotations

import streamlit as st


def render_kpi_panel() -> None:
    """
    Render the KPI scoreboard ribbon.

    Safe to call on every Streamlit rerun.
    Performs no state mutation and no backend calls.
    """

    result = st.session_state.get("last_execution_result")
    if result is None:
        return

    kpis = result.get("kpis", {}) or {}

    # --------------------------------------------------
    # Verbatim values (display-only formatting)
    # --------------------------------------------------
    engine_latency_ms = kpis.get("engine_latency_ms")
    quality_status = kpis.get("quality_status")
    confidence_score = kpis.get("confidence_score")
    task_success = kpis.get("task_success")

    latency_display = (
        f"{engine_latency_ms} ms"
        if engine_latency_ms is not None
        else "N/A"
    )

    confidence_display = (
        str(confidence_score)
        if confidence_score is not None
        else "N/A"
    )

    status_display = (
        str(task_success)
        if task_success is not None
        else "N/A"
    )

    quality_display = quality_status or "N/A"

    # --------------------------------------------------
    # Horizontal KPI ribbon
    # --------------------------------------------------
    with st.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Latency", latency_display)

        with col2:
            st.metric("Answer Quality", quality_display)

        with col3:
            st.metric("Confidence", confidence_display)

        with col4:
            st.metric("Status", status_display)
