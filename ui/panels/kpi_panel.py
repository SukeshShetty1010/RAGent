# === ui/panels/kpi_panel.py ===
"""
KPI Panel (KPI Ribbon)

Responsibility:
- Displays user-facing KPIs for the current execution.

Contract Consumption:
- Reads:
  - `kpis.engine_latency_ms`
  - `kpis.quality_status`
  - `kpis.confidence_score`
  - `kpis.task_success`

State Interactions:
- Reads:
  - `last_execution_result`

Forbidden Actions:
- Must never compute percentiles, averages, or trends.
- Must never read `raw_metrics`.
"""


def render():
    pass