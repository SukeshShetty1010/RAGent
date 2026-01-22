# === ui/app.py ===
"""
UI Application Orchestrator

Responsibility:
- Acts as the top-level coordinator for the Streamlit application.
- Defines the high-level layout and panel composition.
- Controls render order and conditional visibility of panels.

Contract Consumption:
- Reads `ExecutionResult` indirectly via `st.session_state.last_execution_result`.
- Does NOT inspect or mutate any fields inside `ExecutionResult`.

State Interactions:
- Reads:
  - `current_query`
  - `last_execution_result`
  - `is_running`
  - `debug_enabled`
- Writes:
  - None (delegates all state mutation to `query_panel` and `state` utilities).

Forbidden Actions:
- Must never import or call `RageEngine`.
- Must never compute metrics, quality states, or derived values.
- Must never access `raw_metrics` directly.
"""


def run_app():
    pass