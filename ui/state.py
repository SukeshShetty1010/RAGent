# ui/state.py
"""
Session State Definition

Responsibility:
- Defines and documents the canonical Streamlit session state keys.
- Acts as the single source of truth for UI state shape.
- Must be initialized before any panel renders or any backend execution occurs.

Contract Consumption:
- Stores the full `ExecutionResult` object as an **opaque dictionary**.
- Does NOT import, inspect, validate, or depend on backend schemas.

State Interactions:
- Defines and initializes the following keys exactly once:
  - `current_query`: str | None
  - `last_execution_result`: dict | None
  - `is_running`: bool
  - `debug_enabled`: bool

Architectural Guarantees:
- Idempotent: Safe to call on every Streamlit rerun.
- Decoupled: No backend imports, no execution logic.
- Predictable: Default values are stable and explicit.

Forbidden Actions:
- Must never import RageEngine or ExecutionResult.
- Must never mutate the contents of `last_execution_result`.
- Must never compute derived state from backend data.
"""

from typing import Optional, Dict, Any
import streamlit as st


def initialize_state() -> None:
    """
    Initialize Streamlit session state keys if they do not already exist.

    This function must be called at the very top of the application
    lifecycle (e.g., at the start of `app.py`) to guarantee a stable
    and predictable UI state shape.
    """

    if "current_query" not in st.session_state:
        st.session_state.current_query: Optional[str] = None

    if "last_execution_result" not in st.session_state:
        st.session_state.last_execution_result: Optional[Dict[str, Any]] = None

    if "is_running" not in st.session_state:
        st.session_state.is_running: bool = False

    if "debug_enabled" not in st.session_state:
        st.session_state.debug_enabled: bool = False
