# ============================================================
# ui/panels/kpi_panel.py
# KPI Panel — Execution Scoreboard (Tier 1, Capability-Aware)
# ============================================================

from __future__ import annotations

import streamlit as st


def render_kpi_panel() -> None:
    """
    Render the KPI scoreboard ribbon.

    Architectural guarantees:
    - Dumb, read-only component
    - Zero computation or inference
    - Renders backend values verbatim
    - Safe on every Streamlit rerun
    """

    result = st.session_state.get("last_execution_result")
    if result is None:
        return

    kpis = result.get("kpis", {}) or {}

    # --------------------------------------------------
    # Verbatim backend values
    # --------------------------------------------------
    engine_latency_ms = kpis.get("engine_latency_ms")
    quality_status = kpis.get("quality_status")
    confidence_score = kpis.get("confidence_score")
    task_success = kpis.get("task_success")
    answer_capability = kpis.get("answer_capability")

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

    success_display = (
        "Yes" if task_success else "No"
        if task_success is not None
        else "N/A"
    )

    quality_display = quality_status or "N/A"
    capability_display = (
        answer_capability.capitalize()
        if isinstance(answer_capability, str)
        else "N/A"
    )

    # --------------------------------------------------
    # Horizontal KPI ribbon (Tier 1)
    # --------------------------------------------------
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Latency", latency_display)

        with col2:
            st.metric("Evidence Quality", quality_display)

        with col3:
            st.metric("Confidence", confidence_display)

        with col4:
            st.metric("Capability", capability_display)

        with col5:
            st.metric("Completed", success_display)
