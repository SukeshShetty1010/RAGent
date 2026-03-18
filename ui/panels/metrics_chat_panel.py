# ============================================================
# ui/panels/metrics_chat_panel.py
# Conversational Metrics Panel for Chat UI
# ============================================================
"""
This module provides in-chat metric display components
that can be rendered inline with assistant messages.

These are designed to present KPIs and metrics in a
conversational, non-intrusive way that enhances trust
without disrupting the chat flow.
"""

from __future__ import annotations

import streamlit as st
from typing import Dict, List, Any, Optional


# ============================================================
# Metric Card Components
# ============================================================

def render_inline_metrics_card(kpis: Dict[str, Any]) -> None:
    """
    Render a compact inline metrics card for chat messages.
    
    This is designed to appear directly after an assistant message
    providing at-a-glance performance data.
    """
    if not kpis:
        return
    
    latency = kpis.get("engine_latency_ms", 0)
    confidence = kpis.get("confidence_score", 0)
    capability = kpis.get("answer_capability", "unknown")
    quality = kpis.get("quality_status", "unknown")
    
    # Capability emoji
    cap_emoji = {
        "full": "✅",
        "partial": "⚡",
        "insufficient": "⚠️",
    }.get(capability.lower() if capability else "", "❓")
    
    # Quality color
    qual_color = {
        "quality_ok": "#00c853",
        "quality_weak": "#ffc107",
        "quality_empty": "#ff5252",
    }.get(quality.lower() if quality else "", "#888")
    
    st.markdown(
        f"""
        <div style="
            display: flex;
            gap: 16px;
            padding: 8px 12px;
            background: rgba(26, 26, 46, 0.6);
            border-radius: 8px;
            margin-top: 8px;
            font-size: 0.75rem;
            color: #888;
            align-items: center;
        ">
            <span>{cap_emoji} {capability.title() if capability else 'Unknown'}</span>
            <span>⏱️ {int(latency)}ms</span>
            <span style="color: {qual_color}">📊 {int(confidence * 100)}% confidence</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pipeline_stages_compact(stages: List[Dict[str, Any]]) -> None:
    """
    Render pipeline execution stages in a compact timeline format.
    """
    if not stages:
        return
    
    st.markdown(
        """
        <div style="
            padding: 12px;
            background: rgba(26, 26, 46, 0.6);
            border-radius: 8px;
            margin-top: 8px;
        ">
            <div style="font-size: 0.8rem; color: #667eea; margin-bottom: 8px;">
                ⚡ Pipeline Execution
            </div>
        """,
        unsafe_allow_html=True
    )
    
    for stage in stages:
        name = stage.get("name", "unknown")
        status = stage.get("status", "unknown")
        duration = stage.get("duration_ms")
        
        status_emoji = "✓" if status == "completed" else "⏳" if status == "started" else "❌"
        duration_str = f"{int(duration)}ms" if duration else "—"
        
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: space-between;
                padding: 4px 0;
                font-size: 0.7rem;
                color: #aaa;
            ">
                <span>{status_emoji} {name}</span>
                <span>{duration_str}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Full Metrics Report for Sidebar
# ============================================================

def render_sidebar_metrics_report(kpis: Dict[str, Any]) -> None:
    """
    Render a full metrics report in the sidebar.
    """
    if not kpis:
        st.caption("No metrics available yet. Ask a question to see metrics.")
        return
    
    st.markdown("### 📊 Last Query Metrics")
    
    # Primary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "🕐 Latency",
            f"{int(kpis.get('engine_latency_ms', 0))}ms"
        )
        st.metric(
            "📊 Confidence",
            f"{int(kpis.get('confidence_score', 0) * 100)}%"
        )
    
    with col2:
        st.metric(
            "📚 Evidence",
            f"{kpis.get('retrieved_chunks', 0)} chunks"
        )
        llm_latency = kpis.get('llm_latency_ms')
        st.metric(
            "🤖 LLM Time",
            f"{int(llm_latency)}ms" if llm_latency else "—"
        )
    
    # Quality status
    quality = kpis.get("quality_status", "unknown")
    capability = kpis.get("answer_capability", "unknown")
    
    st.markdown(f"**Quality:** `{quality}`")
    st.markdown(f"**Capability:** `{capability}`")
    
    # Success indicator
    success = kpis.get("task_success", False)
    if success:
        st.success("✓ Task completed successfully")
    else:
        st.warning("⚠ Task had limitations")


# ============================================================
# Evidence Summary for Chat
# ============================================================

def render_evidence_summary(evidence: List[Dict[str, Any]], max_sources: int = 3) -> None:
    """
    Render a compact evidence summary for inline display.
    """
    if not evidence:
        return
    
    sources = []
    for item in evidence[:max_sources]:
        title = item.get("source_title") or item.get("source") or "Unknown"
        sources.append(title)
    
    remaining = len(evidence) - max_sources
    
    sources_str = " · ".join(sources)
    if remaining > 0:
        sources_str += f" (+{remaining} more)"
    
    st.markdown(
        f"""
        <div style="
            padding: 8px 12px;
            background: rgba(102, 126, 234, 0.1);
            border-left: 3px solid #667eea;
            border-radius: 4px;
            margin-top: 8px;
            font-size: 0.75rem;
            color: #aaa;
        ">
            📚 <strong>Sources:</strong> {sources_str}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Trust Signal Badge
# ============================================================

def render_trust_badge(capability: str) -> str:
    """
    Return HTML for a trust signal badge.
    """
    badges = {
        "full": """
            <span style="
                display: inline-block;
                padding: 2px 8px;
                background: linear-gradient(135deg, #00c853 0%, #69f0ae 100%);
                color: #000;
                font-size: 0.65rem;
                border-radius: 12px;
                font-weight: 600;
            ">✓ VERIFIED</span>
        """,
        "partial": """
            <span style="
                display: inline-block;
                padding: 2px 8px;
                background: linear-gradient(135deg, #ffc107 0%, #ffeb3b 100%);
                color: #000;
                font-size: 0.65rem;
                border-radius: 12px;
                font-weight: 600;
            ">⚡ EVIDENCE-BOUND</span>
        """,
        "insufficient": """
            <span style="
                display: inline-block;
                padding: 2px 8px;
                background: linear-gradient(135deg, #ff5252 0%, #ff8a80 100%);
                color: #fff;
                font-size: 0.65rem;
                border-radius: 12px;
                font-weight: 600;
            ">⚠ INSUFFICIENT</span>
        """,
    }
    
    return badges.get(capability.lower() if capability else "", badges["partial"])
