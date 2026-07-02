"""
agent/output_validator.py

Structural validation of LLM-generated answers.

This is a post-generation, fail-soft check — it never replaces or
discards `final_answer`. It only annotates `agent_decisions`/`kpis`
with structural compliance signals (unclosed Markdown, missing
required sections for PARTIAL answers) for observability and future
gating.

Streaming caveat: in the streaming engine, tokens are already flushed
to the client before this runs, so validation cannot block or rewrite
already-sent tokens — it only annotates the final done/stage payload.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel

from agent.capability.capability_types import AnswerCapability

REQUIRED_PARTIAL_SECTION = "Unsupported or Missing Parts:"

# Markdown tag characters that must appear in balanced (even) counts.
_MARKDOWN_TAGS = ("**", "_", "`")


class ValidationResult(BaseModel):
    is_valid: bool
    issues: List[str]
    has_required_section: bool


def validate_answer(
    answer: str,
    capability: AnswerCapability,
) -> ValidationResult:
    """
    Deterministic, pure structural validation of a generated answer.
    """
    issues: List[str] = []

    if not answer or not answer.strip():
        issues.append("Answer is empty or whitespace-only")

    for tag in _MARKDOWN_TAGS:
        if answer.count(tag) % 2 != 0:
            issues.append(f"Unclosed Markdown tag: '{tag}'")

    has_required_section = REQUIRED_PARTIAL_SECTION in answer

    if capability == AnswerCapability.PARTIAL and not has_required_section:
        issues.append(
            f"PARTIAL answer missing required section: "
            f"'{REQUIRED_PARTIAL_SECTION}'"
        )

    return ValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        has_required_section=has_required_section,
    )
