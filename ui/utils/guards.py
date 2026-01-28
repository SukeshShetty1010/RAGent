# ui/utils/guards.py
"""
UI Guards and Assertions — Contract Enforcement Layer

Responsibility:
- Acts as the defensive boundary between the backend execution engine
  and the frontend UI panels.
- Validates that the incoming ExecutionResult meets the minimum
  structural contract required for safe rendering.
- Prevents runtime crashes (e.g., KeyError) by detecting malformed data
  early and failing softly.

Architectural Role:
- Pure Python utility module.
- No UI dependencies.
- No backend execution logic.
- Read-only inspection of data.

Contract Consumption:
- Validates the top-level structure of the ExecutionResult dictionary.
- Expected top-level keys:
  - `final_answer`
  - `agent_decisions`
  - `evidence`
  - `kpis`

Failure Philosophy:
- Fail-soft: never raise exceptions.
- Log warnings on contract violations.
- Return a boolean signal to allow the UI to choose a safe fallback state.

Forbidden Actions:
- Must never mutate the result dictionary.
- Must never raise user-facing exceptions.
- Must never import Streamlit or UI components.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def validate_execution_result(result: Any) -> bool:
    """
    Validate the structural integrity of an ExecutionResult-like object.

    Args:
        result: An opaque object expected to be a dictionary produced
                by the backend execution engine.

    Returns:
        True if the result satisfies the minimum required structure.
        False otherwise (with warnings logged).
    """

    if result is None:
        logger.warning("Contract Violation: ExecutionResult is None")
        return False

    if not isinstance(result, Dict):
        logger.warning(
            "Contract Violation: ExecutionResult is not a dict "
            f"(type={type(result)})"
        )
        return False

    required_keys = {
        "final_answer",
        "agent_decisions",
        "evidence",
        "kpis",
    }

    is_valid = True

    for key in required_keys:
        if key not in result:
            logger.warning(
                f"Contract Violation: Missing key '{key}' in ExecutionResult"
            )
            is_valid = False

    return is_valid
