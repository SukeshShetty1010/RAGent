# === ui/panels/evidence_panel.py ===
"""
Evidence Panel

Responsibility:
- Displays supporting evidence chunks and their sources.

Contract Consumption:
- Reads:
  - `evidence`

State Interactions:
- Reads:
  - `last_execution_result`

Forbidden Actions:
- Must never re-rank, filter, or deduplicate evidence.
- Must never compute relevance scores.
"""


def render():
    pass
