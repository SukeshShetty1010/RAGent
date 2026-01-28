# ui/utils/formatting.py
"""
Formatting Utilities — Presentation Layer

Responsibility:
- Provides pure, deterministic formatting helpers for UI display.
- Transforms raw primitive values into human-readable strings.
- Contains no business logic, no state, and no side effects.

Architectural Role:
- Cosmetic layer only.
- Fully isolated from backend and UI frameworks.
- Safe to reuse across panels.

Contract Consumption:
- Accepts only primitive values (int, float, str, bool).
- Does NOT accept or inspect ExecutionResult dictionaries.

Forbidden Actions:
- Must never compute metrics or make judgments.
- Must never access global or shared state.
- Must never import UI or backend modules.
"""

from typing import Optional


def format_latency_ms(val: Optional[float]) -> str:
    """
    Format a latency value in milliseconds.

    Args:
        val: Latency in milliseconds.

    Returns:
        A string formatted as "<int> ms" or "N/A" if unavailable.
    """
    if val is None:
        return "N/A"

    try:
        return f"{int(round(val))} ms"
    except Exception:
        return "N/A"


def format_confidence_score(val: Optional[float]) -> str:
    """
    Format a confidence score as a percentage.

    Args:
        val: Confidence score between 0.0 and 1.0.

    Returns:
        A string formatted as "<int>%" or "N/A" if unavailable.
    """
    if val is None:
        return "N/A"

    try:
        return f"{int(round(val * 100))}%"
    except Exception:
        return "N/A"


def format_quality_status(val: Optional[str]) -> str:
    """
    Prettify a quality status string.

    Args:
        val: Quality status (e.g., "QUALITY_OK").

    Returns:
        A human-readable string (e.g., "Quality Ok") or "N/A".
    """
    if not val:
        return "N/A"

    try:
        return val.replace("_", " ").title()
    except Exception:
        return "N/A"


def format_bool_success(val: Optional[bool]) -> str:
    """
    Format a boolean success indicator.

    Args:
        val: Success flag.

    Returns:
        "Pass", "Fail", or "N/A".
    """
    if val is None:
        return "N/A"

    return "Pass" if val is True else "Fail"
