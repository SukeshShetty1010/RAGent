# === ui/panels/trace_panel.py ===
"""
Trace Panel (Inspector)

Responsibility:
- Displays execution trace and decision metadata for inspection.

Contract Consumption:
- Reads:
  - `agent_decisions`
  - `raw_metrics`

State Interactions:
- Reads:
  - `last_execution_result`
  - `debug_enabled`

Forbidden Actions:
- Must never influence UI behavior or rendering logic.
- Must be strictly read-only.
"""


def render():
    pass
