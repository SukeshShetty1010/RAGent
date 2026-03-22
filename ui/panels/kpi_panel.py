# ============================================================
# ui/panels/kpi_panel.py
# KPI Panel — Executive Trust Snapshot (FINAL)
# ============================================================

from __future__ import annotations

import streamlit as st

from ui.utils.formatting import (
    format_latency_sec,
    format_confidence_score,
    format_quality_status,
    format_answer_confidence,
)


def render_kpi_panel() -> None:
    """
    Render Tier-1 Executive KPI cards.

    Executive guarantees:
    - Trust-first metrics (not grading labels)
    - No error semantics for honest degradation
    - Backend values rendered verbatim
    """

    result = st.session_state.get("last_execution_result")
    if result is None:
        return

    kpis = result.get("kpis", {}) or {}

    # --------------------------------------------------
    # Extract backend KPIs (verbatim)
    # --------------------------------------------------
    latency_ms = kpis.get("engine_latency_ms")
    quality_status = kpis.get("quality_status")
    confidence_score = kpis.get("confidence_score")
    answer_capability = kpis.get("answer_capability")
    task_success = kpis.get("task_success")

    # --------------------------------------------------
    # UI Translation
    # --------------------------------------------------
    latency_display = format_latency_sec(latency_ms)
    quality_display = format_quality_status(quality_status)
    confidence_display = format_confidence_score(confidence_score)

    confidence_meta = format_answer_confidence(answer_capability)
    answer_confidence_label = confidence_meta["label"]

    honesty_display = "100%" if answer_capability in ("full", "partial") else "0%"

    completion_display = "Yes" if task_success else "No"

    # --------------------------------------------------
    # Executive Snapshot
    # --------------------------------------------------
    st.markdown("#### System Snapshot")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="🛡 Honesty",
            value=honesty_display,
            help="System degrades safely instead of hallucinating",
        )

    with col2:
        st.metric(
            label="🎯 Evidence Quality",
            value=quality_display,
            help="Retrieval quality gate outcome",
        )

    with col3:
        st.metric(
            label="📊 Answer Confidence",
            value=answer_confidence_label,
            help="User-facing confidence based on evidence coverage",
        )

    with col4:
        st.metric(
            label="📉 Confidence Score",
            value=confidence_display,
            help="Evidence-derived confidence (normalized)",
        )

    with col5:
        st.metric(
            label="⚡ Latency",
            value=latency_display,
            help="End-to-end execution latency",
        )
