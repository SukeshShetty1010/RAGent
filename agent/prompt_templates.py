# FILE: agent/prompt_templates.py
# Pure Prompt Templates & Formatting (NO OBSERVABILITY)

from __future__ import annotations

from typing import List, Dict, Any

from agent.task_router import TaskType
from agent.capability.capability_types import AnswerCapability
from agent.output_validator import CITATION_FORMAT

# ============================================================
# Core Prompt Assembly
# ============================================================

def construct_final_prompt(
    instruction: str,
    context_block: str,
    query: str,
) -> str:
    guardrails = (
        "CRITICAL FORMATTING: You must properly close all Markdown tags (e.g., **, _, `). "
        "Do not leave dangling formatting characters. Avoid applying bold or italic styling to section headers unless explicitly requested."
    )
    return (
        f"{instruction}\n\n"
        f"{guardrails}\n\n"
        f"=== BEGIN CONTEXT ===\n"
        f"{context_block}\n"
        f"=== END CONTEXT ===\n\n"
        f"=== USER QUERY ===\n"
        f"{query}\n\n"
        f"=== ANSWER ===\n"
    )

# ============================================================
# Context Formatting
# ============================================================

def format_context(
    chunks: List[Dict[str, Any]],
) -> str:
    if not chunks:
        return "No supporting context was retrieved."

    formatted: List[str] = []

    for c in chunks:
        source_title = c.get("source_title") or "Unknown Source"
        source_type = c.get("source_type") or "local"
        content = c.get("content") or ""

        formatted.append(
            f"[Source: {source_title} | Type: {source_type}]\n"
            f"{content}\n"
        )

    return "\n".join(formatted)

# ============================================================
# Instruction Dispatch
# ============================================================

def get_verbose_instruction(
    task: TaskType,
    capability: AnswerCapability,
) -> str:
    if task == TaskType.COMPARISON:
        return comparison_verbose(capability)
    if task == TaskType.LISTICLE:
        return listicle_verbose(capability)
    if task == TaskType.FACTUAL:
        return factual_verbose(capability)
    return open_verbose(capability)


def get_concise_instruction(
    task: TaskType,
    capability: AnswerCapability,
) -> str:
    if task == TaskType.COMPARISON:
        return comparison_concise(capability)
    if task == TaskType.LISTICLE:
        return listicle_concise(capability)
    if task == TaskType.FACTUAL:
        return factual_concise(capability)
    return open_concise(capability)

# ============================================================
# Instruction Templates
# ============================================================

# ------------------------
# COMPARISON
# ------------------------

def comparison_verbose(capability: AnswerCapability) -> str:
    base = (
        "Compare entities across these dimensions:\n"
        "1. Gameplay\n"
        "2. Story\n"
        "3. World Design\n"
        "4. Tone\n"
        "5. Systems\n\n"
        "Use only the provided context.\n"
        f"Cite every factual claim inline as {CITATION_FORMAT}.\n"
    )

    if capability == AnswerCapability.PARTIAL:
        base += (
            "\nIMPORTANT:\n"
            "Explicitly state what information is missing instead of guessing.\n"
            "Always append missing parameters under a separate section titled 'Unsupported or Missing Parts:'.\n"
        )

    return base


def comparison_concise(capability: AnswerCapability) -> str:
    base = (
        "Compare entities by Gameplay, Story, World Design, "
        "Tone, and Systems using only the context. "
        "Cite inline as (Source: 'Title')."
    )

    if capability == AnswerCapability.PARTIAL:
        base += " Explicitly note missing information under 'Unsupported or Missing Parts:'."

    return base

# ------------------------
# LISTICLE
# ------------------------

def listicle_verbose(capability: AnswerCapability) -> str:
    base = (
        "Produce an ordered list strictly from the provided context.\n"
        "Preserve original ordering when present.\n"
        "Do not invent items.\n"
        f"Cite each item inline as {CITATION_FORMAT}.\n"
    )

    if capability == AnswerCapability.PARTIAL:
        base += (
            "\nIMPORTANT:\n"
            "Clearly state that the list is partial if incomplete.\n"
            "Always append missing parameters under a separate section titled 'Unsupported or Missing Parts:'.\n"
        )

    return base


def listicle_concise(capability: AnswerCapability) -> str:
    base = "Create an ordered list from context only. Cite inline as (Source: 'Title')."

    if capability == AnswerCapability.PARTIAL:
        base += " Note that the list may be incomplete under 'Unsupported or Missing Parts:'."

    return base

# ------------------------
# FACTUAL
# ------------------------

def factual_verbose(capability: AnswerCapability) -> str:
    base = (
        "Answer concisely using only the provided context.\n"
        "Limit the response to factual information.\n"
        f"Cite every factual claim inline as {CITATION_FORMAT}.\n"
    )

    if capability == AnswerCapability.PARTIAL:
        base += (
            "\nIMPORTANT:\n"
            "Explicitly state missing information instead of guessing.\n"
            "Always append missing parameters under a separate section titled 'Unsupported or Missing Parts:'.\n"
        )

    return base


def factual_concise(capability: AnswerCapability) -> str:
    base = "Answer factually from the context. Cite inline as (Source: 'Title')."

    if capability == AnswerCapability.PARTIAL:
        base += " Note missing information explicitly under 'Unsupported or Missing Parts:'."

    return base

# ------------------------
# OPEN
# ------------------------

def open_verbose(capability: AnswerCapability) -> str:
    base = (
        "Answer using only the provided context.\n"
        "Organize clearly by topic.\n"
        f"Cite every factual claim inline as {CITATION_FORMAT}.\n"
    )

    if capability == AnswerCapability.PARTIAL:
        base += (
            "\nIMPORTANT:\n"
            "Clearly state any unsupported or missing parts.\n"
            "Always append missing parameters under a separate section titled 'Unsupported or Missing Parts:'.\n"
        )

    return base


def open_concise(capability: AnswerCapability) -> str:
    base = "Answer using the context. Cite inline as (Source: 'Title')."

    if capability == AnswerCapability.PARTIAL:
        base += " State any limitations clearly under 'Unsupported or Missing Parts:'."

    return base

# ============================================================
# Insufficient Prompt
# ============================================================

def insufficient_prompt(query: str) -> str:
    return (
        "The system does not have sufficient reliable information "
        "to answer the following request safely.\n\n"
        f"USER QUERY:\n{query}\n\n"
        "Respond with a brief, honest refusal explaining that "
        "the available evidence is insufficient."
    )
