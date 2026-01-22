# === ui/panels/query_panel.py ===
"""
Query Panel

Responsibility:
- Captures user input (query text).
- Initiates execution requests via external controller logic.

Contract Consumption:
- None (input-only panel).

State Interactions:
- Reads:
  - `is_running`
- Writes:
  - `current_query`
  - `is_running`

Forbidden Actions:
- Must never read `ExecutionResult`.
- Must never render answers, evidence, or KPIs.
"""


def render():
    pass