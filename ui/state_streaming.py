# ============================================================
# ui/state_streaming.py
# Session State for Conversational Streaming UI
# ============================================================
"""
Session state management for the streaming conversational UI.

State Keys:
- chat_history: List of message dicts with role, content, timestamp, metadata
- is_processing: Boolean flag for active query processing
- pending_query: Query from sidebar/quick buttons waiting to be processed
- last_result: Full ExecutionResult from the most recent query
- show_metrics: User preference for metrics panel visibility
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import streamlit as st


def initialize_streaming_state() -> None:
    """
    Initialize Streamlit session state keys for streaming chat UI.
    
    This function must be called at the top of the application lifecycle
    before any component renders.
    """
    
    # Chat history: list of message objects
    # Each message: { role: str, content: str, timestamp: float, metadata: dict }
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, Any]] = []
    
    # Processing flag: prevents duplicate submissions
    if "is_processing" not in st.session_state:
        st.session_state.is_processing: bool = False
    
    # Pending query: set by sidebar quick buttons
    if "pending_query" not in st.session_state:
        st.session_state.pending_query: Optional[str] = None
    
    # Last execution result: for metrics and evidence panels
    if "last_result" not in st.session_state:
        st.session_state.last_result: Optional[Dict[str, Any]] = None
    
    # UI preferences (persisted across reruns)
    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics: bool = True
    
    if "show_evidence" not in st.session_state:
        st.session_state.show_evidence: bool = True
    
    if "show_debug" not in st.session_state:
        st.session_state.show_debug: bool = False


def clear_chat_state() -> None:
    """Reset all chat-related state."""
    st.session_state.chat_history = []
    st.session_state.is_processing = False
    st.session_state.pending_query = None
    st.session_state.last_result = None


def get_chat_history() -> List[Dict[str, Any]]:
    """Get the current chat history."""
    return st.session_state.get("chat_history", [])


def get_last_result() -> Optional[Dict[str, Any]]:
    """Get the last execution result."""
    return st.session_state.get("last_result")


def set_processing(status: bool) -> None:
    """Set the processing status."""
    st.session_state.is_processing = status
