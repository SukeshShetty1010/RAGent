# === ui/panels/answer_panel.py ===
"""
Answer Panel (Answer Card)

Responsibility:
- Renders the final answer verbatim to the user.

Contract Consumption:
- Reads:
  - `final_answer`
  - `kpis.quality_status`

State Interactions:
- Reads:
  - `last_execution_result`

Forbidden Actions:
- Must never modify or summarize `final_answer`.
- Must never infer success or failure beyond `quality_status`.
- Must never access `raw_metrics`.
"""


def render():
    pass