# === ui/state.py ===
"""
Session State Definition

Responsibility:
- Defines and documents the canonical Streamlit session state keys.
- Acts as the single source of truth for UI state shape.

Contract Consumption:
- Stores the full `ExecutionResult` object without modification.

State Interactions:
- Defines the following keys:
  - `current_query`: str | None
  - `last_execution_result`: dict | None (ExecutionResult)
  - `is_running`: bool
  - `debug_enabled`: bool

Forbidden Actions:
- Must never inspect or transform `ExecutionResult`.
- Must never compute derived UI state from backend data.
"""

def initialize_state():
    pass