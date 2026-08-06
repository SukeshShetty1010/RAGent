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

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agent.capability.capability_types import AnswerCapability

REQUIRED_PARTIAL_SECTION = "Unsupported or Missing Parts:"

# Markdown tag characters that must appear in balanced (even) counts.
_MARKDOWN_TAGS = ("**", "_", "`")

# ------------------------------------------------------------------
# Citation contract (single source of truth).
#
# Prompt templates (agent/prompt_templates.py) instruct the LLM to cite
# in this exact format, and tests/evaluation_metrics.py imports
# CITATION_PATTERN rather than keeping its own copy — the two drifting
# apart is what silently broke citation-grounding measurement before.
# ------------------------------------------------------------------
CITATION_FORMAT = "(Source: 'Exact Source Title')"
CITATION_PATTERN = re.compile(r"\(Source:\s*'([^']+)'\)")


class ValidationResult(BaseModel):
    is_valid: bool
    issues: List[str]
    has_required_section: bool
    cited_sources: List[str] = []
    unmatched_citations: List[str] = []
    citation_count: int = 0


def _normalize(text: str) -> str:
    return text.lower().strip()


def _titles_match(cited: str, context_title: str) -> bool:
    cited_n = _normalize(cited)
    context_n = _normalize(context_title)
    return (
        cited_n == context_n
        or cited_n in context_n
        or context_n in cited_n
    )


def validate_answer(
    answer: str,
    capability: AnswerCapability,
    *,
    context_chunks: Optional[List[Dict[str, Any]]] = None,
) -> ValidationResult:
    """
    Deterministic, pure structural validation of a generated answer.

    When `context_chunks` is supplied, also validates that every cited
    source (format: CITATION_FORMAT) matches a `source_title` present
    in the assembled context. Citations not backed by context are
    flagged in `unmatched_citations` and `issues`, but — consistent
    with the fail-soft contract of this module — never alter or
    discard `answer` itself.
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

    cited_sources = CITATION_PATTERN.findall(answer)
    unmatched_citations: List[str] = []

    if context_chunks is not None:
        context_titles = [
            c.get("source_title")
            for c in context_chunks
            if c.get("source_title")
        ]

        for cited in cited_sources:
            if not any(_titles_match(cited, t) for t in context_titles):
                unmatched_citations.append(cited)

        if unmatched_citations:
            issues.append(
                f"Cited source(s) not found in retrieved context: "
                f"{unmatched_citations}"
            )

        if answer.strip() and not cited_sources:
            issues.append(
                f"Answer contains no citations in the required format "
                f"{CITATION_FORMAT}"
            )

    return ValidationResult(
        is_valid=len(issues) == 0,
        issues=issues,
        has_required_section=has_required_section,
        cited_sources=cited_sources,
        unmatched_citations=unmatched_citations,
        citation_count=len(cited_sources),
    )
