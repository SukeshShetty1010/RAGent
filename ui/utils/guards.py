# ============================================================
# ui/utils/guards.py
# UI Guards — ExecutionResult Contract Validation
# ============================================================

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def validate_execution_result(result: Any) -> bool:
    """
    Validate the minimum structural integrity of an ExecutionResult.

    This guard protects UI panels from malformed or partial backend data.

    Design principles:
    - Fail-soft (never raise)
    - Log warnings for observability
    - Enforce only MINIMUM required structure
    """

    # --------------------------------------------------
    # Basic existence & type checks
    # --------------------------------------------------
    if result is None:
        logger.warning(
            "UI Guard: ExecutionResult is None — nothing to render"
        )
        return False

    if not isinstance(result, dict):
        logger.warning(
            "UI Guard: ExecutionResult must be a dict "
            f"(received type={type(result)})"
        )
        return False

    # --------------------------------------------------
    # Required top-level keys (minimal contract)
    # --------------------------------------------------
    required_keys = {
        "final_answer",
        "agent_decisions",
        "evidence",
        "kpis",
    }

    missing_keys = [
        key for key in required_keys if key not in result
    ]

    if missing_keys:
        logger.warning(
            "UI Guard: ExecutionResult missing required keys: "
            + ", ".join(missing_keys)
        )
        return False

    # --------------------------------------------------
    # Shallow type sanity checks (NO deep validation)
    # --------------------------------------------------
    if not isinstance(result.get("agent_decisions"), dict):
        logger.warning(
            "UI Guard: 'agent_decisions' is not a dict"
        )
        return False

    if not isinstance(result.get("kpis"), dict):
        logger.warning(
            "UI Guard: 'kpis' is not a dict"
        )
        return False

    if not isinstance(result.get("evidence"), list):
        logger.warning(
            "UI Guard: 'evidence' is not a list"
        )
        return False

    # --------------------------------------------------
    # Contract satisfied
    # --------------------------------------------------
    return True
