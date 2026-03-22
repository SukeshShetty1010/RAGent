# ============================================================
# ui/app_streaming.py
# Streamlit Chat UI with Per-Response Metrics
# ============================================================
from __future__ import annotations

import sys
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import streamlit as st

# Add project root to Python path for module resolution
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.execution_engine_streaming import StreamingRageEngine, StreamingStage
from ui.state_streaming import clear_chat_state, initialize_streaming_state


CHAT_CSS = """
<style>
    :root {
        --bg-soft: #020817;
        --panel: #0b1220;
        --panel-2: #111a2b;
        --line: #233146;
        --ink: #e5e7eb;
        --muted: #93a3b8;
        --brand: #22d3ee;
        --brand-soft: #164e63;
        --accent: #fb923c;
        --ok: #22c55e;
        --warn: #f59e0b;
        --bad: #ef4444;
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, #0f172a 0%, transparent 35%),
            radial-gradient(circle at 95% 5%, #111827 0%, transparent 30%),
            var(--bg-soft);
    }
    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }
    .app-title {
        color: var(--brand);
        font-weight: 800;
        letter-spacing: 0.1px;
        margin-bottom: 0.15rem;
    }
    .app-subtitle {
        color: var(--muted);
        margin-bottom: 1rem;
    }
    .assistant-title {
        color: var(--brand);
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }
    .assistant-box {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 6px 18px rgba(2, 6, 23, 0.35);
    }
    .assistant-box p {
        color: var(--ink);
        line-height: 1.65;
        font-size: 0.98rem;
        margin: 0.28rem 0;
    }
    .assistant-box h3 {
        color: #67e8f9;
        margin-top: 0.7rem;
        margin-bottom: 0.2rem;
        font-size: 1rem;
    }
    .metric-strip {
        background: linear-gradient(90deg, #082f49 0%, #1f2937 100%);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 9px 12px;
        margin-top: 8px;
        margin-bottom: 8px;
        color: #e2e8f0;
        font-weight: 600;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        margin-bottom: 6px;
    }
    .kpi-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px;
        min-height: 68px;
    }
    .kpi-label {
        color: var(--muted);
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .kpi-value {
        color: var(--ink);
        font-weight: 700;
        margin-top: 3px;
        font-size: 1.12rem;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 8px;
    }
    .chip {
        font-size: 0.74rem;
        border-radius: 999px;
        padding: 4px 9px;
        border: 1px solid var(--line);
        background: var(--panel-2);
        color: #cbd5e1;
    }
    .chip-ok { border-color: #166534; background: #052e16; color: #86efac; }
    .chip-warn { border-color: #9a3412; background: #431407; color: #fdba74; }
    .chip-bad { border-color: #991b1b; background: #450a0a; color: #fca5a5; }
    .evidence-card {
        background: #0b1220;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 9px 11px;
        margin-bottom: 8px;
    }
    .empty-state {
        margin-top: 1rem;
        border: 1px dashed var(--line);
        border-radius: 12px;
        padding: 1rem;
        color: var(--muted);
        background: rgba(15, 23, 42, 0.35);
    }
    [data-testid="stChatMessage"] {
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }
    [data-testid="stChatInput"] {
        background: rgba(2, 6, 23, 0.65);
        border-top: 1px solid var(--line);
        backdrop-filter: blur(6px);
    }
    @media (max-width: 920px) {
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
</style>
"""


@st.cache_resource
def get_engine() -> StreamingRageEngine:
    return StreamingRageEngine()


def add_message(role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    st.session_state.chat_history.append(
        {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
    )


def _as_label(name: str) -> str:
    return name.replace("_", " ").title()


def _as_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if 0.0 <= value <= 1.0:
            return f"{value:.2%}"
        return f"{value:.2f}"
    return str(value)


def _extract_urls(text: str) -> List[str]:
    urls = re.findall(r"https?://[^\s)>\]]+", text or "")
    deduped: List[str] = []
    seen = set()
    for u in urls:
        clean = u.rstrip(".,;:!?")
        if clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def _strip_source_noise(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\[Source:[^\]]*\]\((https?://[^)]+)\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[Source:[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Source:\s*[^\n\[]*(?:\[[^\]]*\])?(?:\((https?://[^)]+)\))?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\|\s*Type:\s*[^\]\n]+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*https?://[^)]+\s*\)", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _prettify_answer(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return "No answer generated."

    # 1. UI Routing: Split "Unsupported or Missing Parts"
    import re
    split_pattern = r"(?i)\n?\s*\**Unsupported\s*(?:or)?\s*Missing\s*Parts?:?\**\s*\n?"
    parts = re.split(split_pattern, text, maxsplit=1)
    if len(parts) > 1:
        text = parts[0].strip()

    # 2. Formatting Sanitization
    text = re.sub(r'(?i)\*\*unsupported or missing parts\*\*', '', text)
    text = re.sub(r'(?<!\*)\*\*(?!\*)', '', text)
    text = re.sub(r'(?<!_)__(?!_)', '', text)

    text = _strip_source_noise(text)

    # Remove isolated markdown artifacts
    text = re.sub(r"(?m)^\s*\*\s*$", "", text)

    # Fix section labels like "Gameplay** ..." -> "**Gameplay** ..."
    text = re.sub(
        r"(?m)(?<!\*)\b(Gameplay|Story|World Design|Tone|Systems)\*\*",
        r"**\1**",
        text,
    )

    # Convert "Gameplay*" / "Story*" style labels to headings
    text = re.sub(
        r"(?m)^\s*(Gameplay|Story|World Design|Tone|Systems)\*?\s*$",
        r"### \1",
        text,
    )

    text = re.sub(
        r"(?m)^\s*(Gameplay|Story|World Design|Tone|Systems)\*?\s*",
        r"### \1\n",
        text,
    )

    # Convert patterns like "* Section:*" into markdown headings
    text = re.sub(
        r"\s*\*\s*([A-Za-z][A-Za-z0-9 /&'-]{2,50}):\s*\*",
        r"\n\n### \1",
        text,
    )

    # Normalize star separators between sentences to readable breaks
    text = re.sub(r"\s+\*\s+", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _render_sources_popover(urls: List[str]) -> None:
    if not urls:
        return
    with st.popover("ℹ Sources"):
        for i, url in enumerate(urls, start=1):
            host = urlparse(url).netloc or "source"
            st.markdown(f"{i}. [{host}]({url})")


def _render_assistant_markdown(target: Any, answer: str, cursor: bool = False) -> List[str]:
    urls = _extract_urls(answer)
    rendered = _prettify_answer(answer)
    if cursor:
        rendered = f"{rendered}\n\n▌"
    target.markdown(rendered)
    return urls


def _quality_chip_class(quality_status: str) -> str:
    q = (quality_status or "").lower()
    if "ok" in q:
        return "chip-ok"
    if "weak" in q:
        return "chip-warn"
    return "chip-bad"


def _capability_chip_class(capability: str) -> str:
    c = (capability or "").lower()
    if c == "full":
        return "chip-ok"
    if c == "partial":
        return "chip-warn"
    return "chip-bad"


def render_message_metrics(metadata: Dict[str, Any], index: int) -> None:
    kpis = metadata.get("kpis", {})
    raw_metrics = metadata.get("raw_metrics", {})
    stages = metadata.get("stages", [])
    evidence = metadata.get("evidence", [])
    agent_decisions = metadata.get("agent_decisions", {})

    if not kpis and not raw_metrics and not stages:
        return

    show_metrics = bool(st.session_state.get("show_metrics", True))
    show_evidence = bool(st.session_state.get("show_evidence", True))
    show_debug = bool(st.session_state.get("show_debug", False))

    if not any([show_metrics, show_evidence, show_debug]):
        return

    st.markdown('<div class="metric-strip">Response metrics</div>', unsafe_allow_html=True)

    if show_metrics and kpis:
        latency_ms = kpis.get("engine_latency_ms")
        llm_latency_ms = kpis.get("llm_latency_ms")
        confidence = kpis.get("confidence_score")
        chunks = kpis.get("retrieved_chunks")
        quality = str(kpis.get("quality_status", "unknown"))
        capability = str(kpis.get("answer_capability", "unknown"))
        success = bool(kpis.get("task_success"))

        latency_display = f"{float(latency_ms)/1000.0:.2f} sec" if latency_ms is not None else "—"
        llm_display = f"{float(llm_latency_ms)/1000.0:.2f} sec" if llm_latency_ms is not None else "—"
        confidence_display = (
            f"{float(confidence) * 100:.1f}%" if confidence is not None else "—"
        )
        chunks_display = str(chunks) if chunks is not None else "—"

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card"><div class="kpi-label">Engine Latency</div><div class="kpi-value">{latency_display}</div></div>
                <div class="kpi-card"><div class="kpi-label">LLM Latency</div><div class="kpi-value">{llm_display}</div></div>
                <div class="kpi-card"><div class="kpi-label">Confidence</div><div class="kpi-value">{confidence_display}</div></div>
                <div class="kpi-card"><div class="kpi-label">Evidence Chunks</div><div class="kpi-value">{chunks_display}</div></div>
            </div>
            <div class="chip-row">
                <span class="chip {_quality_chip_class(quality)}">Quality: {quality}</span>
                <span class="chip {_capability_chip_class(capability)}">Capability: {capability}</span>
                <span class="chip {'chip-ok' if success else 'chip-warn'}">Task Success: {success}</span>
                <span class="chip {'chip-ok' if kpis.get('llm_ran') else 'chip-bad'}">LLM Ran: {bool(kpis.get('llm_ran'))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if show_metrics:
        overview_tab, kpi_json_tab, raw_tab = st.tabs(["Pipeline", "KPI JSON", "Raw Metrics"])
        with overview_tab:
            if stages:
                stage_rows = []
                for stage in stages:
                    stage_rows.append(
                        {
                            "name": stage.get("name", "unknown"),
                            "status": stage.get("status", "unknown"),
                            "duration_ms": round(stage.get("duration_ms", 0.0), 2)
                            if stage.get("duration_ms") is not None
                            else None,
                        }
                    )
                st.dataframe(stage_rows, width="stretch")
            else:
                st.caption(f"No stage data for message #{index}.")
        with kpi_json_tab:
            st.json(kpis if kpis else {"info": "No KPI fields available"})
        with raw_tab:
            st.json(raw_metrics if raw_metrics else {"info": "No raw metrics available"})

    if show_evidence:
        with st.expander(f"Evidence ({len(evidence)})", expanded=False):
            if not evidence:
                st.caption("No evidence returned for this response.")
            for ev_idx, item in enumerate(evidence, start=1):
                source = item.get("source_title") or item.get("source") or "Unknown Source"
                source_type = item.get("source_type", "local")
                content = item.get("content", "")
                preview = f"{content[:350]}..." if len(content) > 350 else content
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <strong>[{ev_idx}] {source}</strong><br/>
                        <span style="color:#64748b;font-size:0.78rem;">{source_type}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if preview:
                    st.caption(preview)

    if show_debug:
        with st.expander("Agent decisions", expanded=False):
            st.json(agent_decisions if agent_decisions else {"info": "No decision metadata"})


def render_chat_history() -> None:
    for i, message in enumerate(st.session_state.chat_history, start=1):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown('<div class="assistant-title">RAGent Response</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    urls = _render_assistant_markdown(st, message["content"], cursor=False)
                _render_sources_popover(urls)
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant":
                render_message_metrics(message.get("metadata", {}), i)


def process_latest_user_message() -> None:
    if not st.session_state.chat_history:
        return

    last_message = st.session_state.chat_history[-1]
    if last_message["role"] != "user":
        st.session_state.is_processing = False
        return

    query = last_message["content"]
    stage_names = {
        "routing": "Routing query",
        "strategy": "Selecting retrieval strategy",
        "retrieval": "Retrieving evidence",
        "capability": "Assessing capability",
        "context_assembly": "Assembling context",
        "prompt_construction": "Constructing prompt",
        "generation": "Generating response",
    }

    with st.chat_message("assistant"):
        status = st.status("Processing request...", expanded=True)
        stream_placeholder = st.empty()
        chunks: List[str] = []

        def on_stage(stage: StreamingStage) -> None:
            if stage.status == "started":
                label = stage_names.get(stage.name, stage.name)
                status.update(label=f"{label}...", state="running")

        def on_token(token: str) -> None:
            chunks.append(token)
            _render_assistant_markdown(stream_placeholder, "".join(chunks), cursor=True)
            time.sleep(0.008)

        engine = get_engine()
        result = engine.run_streaming(
            query=query,
            on_token_callback=on_token,
            on_stage_callback=on_stage,
        )

        # Guarantee visible streaming even when backend returns one large chunk.
        # If true token streaming happened, keep the final render only.
        if len(chunks) <= 2:
            words = result.final_answer.split()
            progressive: List[str] = []
            for word in words:
                progressive.append(word)
                _render_assistant_markdown(stream_placeholder, " ".join(progressive), cursor=True)
                time.sleep(0.025)

        # Ensure we only print the raw output ONCE to the terminal
        print("\n=== TERMINAL RAW OUTPUT ===")
        print(result.final_answer)
        print("===========================\n")

        _render_assistant_markdown(stream_placeholder, result.final_answer, cursor=False)
        _render_sources_popover(_extract_urls(result.final_answer))

        status.update(label="Completed", state="complete")

        metadata = {
            "kpis": result.kpis,
            "raw_metrics": result.raw_metrics,
            "evidence": result.evidence,
            "agent_decisions": result.agent_decisions,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                }
                for s in result.stages
            ],
        }

        render_message_metrics(metadata, len(st.session_state.chat_history) + 1)
        add_message("assistant", result.final_answer, metadata)
        st.session_state.last_result = {
            "final_answer": result.final_answer,
            "kpis": result.kpis,
            "raw_metrics": result.raw_metrics,
            "evidence": result.evidence,
            "agent_decisions": result.agent_decisions,
        }
        st.session_state.is_processing = False


def run_streaming_app() -> None:
    st.set_page_config(
        page_title="RAGent Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_streaming_state()
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    st.markdown('<h1 class="app-title">RAGent Chat</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Evidence-bound answers with per-response metrics</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Chat Controls")
        st.session_state.show_metrics = st.checkbox(
            "Show metrics per response",
            value=st.session_state.show_metrics,
        )
        st.session_state.show_evidence = st.checkbox(
            "Show evidence sections",
            value=st.session_state.show_evidence,
        )
        st.session_state.show_debug = st.checkbox(
            "Show agent decision sections",
            value=st.session_state.show_debug,
        )

        if st.button("Clear chat", width="stretch"):
            clear_chat_state()
            st.rerun()

        st.divider()
        st.caption("Quick prompts")
        samples = {
            "Game comparison": "Compare Far Cry 5 vs Assassin's Creed Valhalla",
            "Release info": "What is the release date of Far Cry 5?",
            "Top list": "Top 5 things to do in Far Cry 5",
            "Recent updates": "Latest update for Assassin's Creed Valhalla",
        }
        for label, prompt in samples.items():
            if st.button(label, width="stretch"):
                st.session_state.pending_query = prompt
                st.rerun()

    render_chat_history()
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-state">
                Ask a question to start the conversation. Responses will stream live and include
                metrics, evidence, and pipeline timing.
            </div>
            """,
            unsafe_allow_html=True,
        )

    query_to_submit: Optional[str] = None
    if st.session_state.pending_query and not st.session_state.is_processing:
        query_to_submit = st.session_state.pending_query
        st.session_state.pending_query = None

    user_input = st.chat_input(
        "Ask RAGent anything about your indexed game data...",
        disabled=st.session_state.is_processing,
    )
    if user_input and not st.session_state.is_processing:
        query_to_submit = user_input

    if query_to_submit:
        add_message("user", query_to_submit)
        st.session_state.is_processing = True
        st.rerun()

    if st.session_state.is_processing:
        process_latest_user_message()
        st.rerun()


if __name__ == "__main__":
    run_streaming_app()
