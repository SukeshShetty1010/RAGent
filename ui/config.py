# === ui/config.py ===
"""
UI Configuration Constants

Responsibility:
- Defines static UI configuration values (titles, defaults, feature flags).
- Centralizes non-dynamic UI constants.

Contract Consumption:
- None (configuration only).

State Interactions:
- None.

Forbidden Actions:
- Must never reference `ExecutionResult`.
- Must never contain environment-specific logic.
"""


APP_TITLE = "RAGent"
DEFAULT_QUERY = ""
DEBUG_DEFAULT = False