# ============================================================
# ui/utils/formatting.py
# Formatting Utilities — Presentation Layer (FINAL)
# ============================================================

from typing import Optional, Dict


# ------------------------------------------------------------
# Latency
# ------------------------------------------------------------
def format_latency_ms(val: Optional[float]) -> str:
    if val is None:
        return "—"
    try:
        return f"{int(round(val))} ms"
    except Exception:
        return "—"


# ------------------------------------------------------------
# Confidence / Percentages
# ------------------------------------------------------------
def format_confidence_score(val: Optional[float]) -> str:
    """
    Backend confidence score: 0.0–1.0
    UI display: percentage
    """
    if val is None:
        return "—"
    try:
        return f"{int(round(val * 100))}%"
    except Exception:
        return "—"


def format_ratio(val: Optional[float]) -> str:
    if val is None:
        return "—"
    try:
        return f"{round(val * 100, 1)}%"
    except Exception:
        return "—"


# ------------------------------------------------------------
# ✅ Answer Confidence (UI Translation Layer)
# ------------------------------------------------------------
CAPABILITY_UI_MAP: Dict[str, Dict[str, str]] = {
    "full": {
        "label": "Verified Answer",
        "description": "This response is fully supported by retrieved evidence.",
        "tone": "success",
    },
    "partial": {
        "label": "Evidence-Bound Answer",
        "description": (
            "Only claims supported by available evidence are included. "
            "Unverifiable aspects were intentionally omitted."
        ),
        "tone": "info",
    },
    "insufficient": {
        "label": "Insufficient Evidence",
        "description": (
            "No reliable sources were found to answer this safely. "
            "The system avoided speculation or hallucination."
        ),
        "tone": "neutral",
    },
}


def format_answer_confidence(val: Optional[str]) -> Dict[str, str]:
    """
    Translate internal answer_capability → user-facing confidence language.

    Returns:
        {
            label: str,
            description: str,
            tone: str
        }
    """
    if not val:
        return {
            "label": "Unknown",
            "description": "Answer confidence could not be determined.",
            "tone": "neutral",
        }

    return CAPABILITY_UI_MAP.get(
        val.lower(),
        {
            "label": val.upper(),
            "description": "Unrecognized answer confidence state.",
            "tone": "neutral",
        },
    )


# ------------------------------------------------------------
# Quality Status
# ------------------------------------------------------------
def format_quality_status(val: Optional[str]) -> str:
    if not val:
        return "—"
    try:
        return val.replace("_", " ").title()
    except Exception:
        return "—"


# ------------------------------------------------------------
# Boolean Outcomes
# ------------------------------------------------------------
def format_bool_success(val: Optional[bool]) -> str:
    if val is None:
        return "—"
    return "Yes" if val is True else "No"
