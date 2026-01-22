# === ui/utils/guards.py ===
"""
UI Guards and Assertions

Responsibility:
- Defines defensive checks to enforce UI contract invariants.

Contract Consumption:
- Validates presence or absence of expected fields without mutation.

State Interactions:
- Reads:
  - `last_execution_result`

Forbidden Actions:
- Must never alter state or data.
- Must never raise user-facing exceptions.
"""

def validate_execution_result(result):
    pass