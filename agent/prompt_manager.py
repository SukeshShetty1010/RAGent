# FILE: agent/prompt_manager.py
# Capability-Aware Prompt Budgeting Engine (ORCHESTRATION ONLY)

from __future__ import annotations

import logging
from typing import List, Dict, Any

from agent.task_router import TaskType
from agent.capability.capability_types import AnswerCapability
from agent.prompt_templates import (
    format_context,
    construct_final_prompt,
    insufficient_prompt,
    get_verbose_instruction,
    get_concise_instruction,
)

from tests.observability import ProfileBlock, MetricsRegistry

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("RAG_PROMPT_MANAGER")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============================================================
# Constants
# ============================================================

# HARD safety cap for the FINAL serialized prompt string
MAX_TOTAL_PROMPT_CHARS = 4500

# ============================================================
# Prompt Manager
# ============================================================

class PromptManager:
    """
    Capability-aware prompt construction engine.

    Orchestration responsibilities only:
    - Prompt budget enforcement
    - Capability guardrails
    - Observability & metrics
    - Delegation to prompt_templates for all text logic
    """

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def generate_prompt(
        self,
        *,
        query: str,
        chunks: List[Dict[str, Any]],
        task: TaskType,
        capability: AnswerCapability,
    ) -> str:
        """
        Generate a prompt that respects:
        - TaskType (structure)
        - AnswerCapability (honesty constraints)
        - MAX_TOTAL_PROMPT_CHARS (hard budget)
        """

        with ProfileBlock("PromptConstruction"):

            # ------------------------------------------------
            # Capability guardrail (SAFE REFUSAL)
            # ------------------------------------------------
            if capability == AnswerCapability.INSUFFICIENT:
                registry = MetricsRegistry.get()
                registry.record("prompt_mode", "insufficient")
                registry.record(
                    "prompt_budget_mode",
                    "insufficient_safe_refusal",
                )
                return insufficient_prompt(query)

            # ------------------------------------------------
            # Context formatting (highest-value payload)
            # ------------------------------------------------
            with ProfileBlock("ContextFormatting"):
                context_chunks = list(chunks)
                context_block = format_context(context_chunks)

            # ------------------------------------------------
            # Instruction selection (task + capability)
            # ------------------------------------------------
            verbose_instruction = get_verbose_instruction(
                task, capability
            )
            concise_instruction = get_concise_instruction(
                task, capability
            )

            # =================================================
            # ATTEMPT 1 — VERBOSE + FULL CONTEXT
            # =================================================
            prompt = construct_final_prompt(
                verbose_instruction,
                context_block,
                query,
            )

            if len(prompt) <= MAX_TOTAL_PROMPT_CHARS:
                MetricsRegistry.get().record(
                    "prompt_budget_mode", "verbose"
                )
                return prompt

            logger.warning("📉 Prompt overflow — switching to concise mode")

            # =================================================
            # FALLBACK A — CONCISE + FULL CONTEXT
            # =================================================
            prompt = construct_final_prompt(
                concise_instruction,
                context_block,
                query,
            )

            if len(prompt) <= MAX_TOTAL_PROMPT_CHARS:
                MetricsRegistry.get().record(
                    "prompt_budget_mode", "concise"
                )
                return prompt

            logger.warning(
                "📉 Prompt still too large — truncating context tail"
            )

            # =================================================
            # FALLBACK B — CONCISE + TRUNCATED CONTEXT
            # =================================================
            while context_chunks:
                context_chunks.pop()
                context_block = format_context(context_chunks)

                prompt = construct_final_prompt(
                    concise_instruction,
                    context_block,
                    query,
                )

                if len(prompt) <= MAX_TOTAL_PROMPT_CHARS:
                    MetricsRegistry.get().record(
                        "prompt_budget_mode", "truncated"
                    )
                    MetricsRegistry.get().observe(
                        "prompt_context_chunks_used",
                        len(context_chunks),
                    )
                    return prompt

            # ------------------------------------------------
            # ABSOLUTE FAIL-SAFE (MINIMAL PROMPT)
            # ------------------------------------------------
            MetricsRegistry.get().record(
                "prompt_budget_mode", "minimal"
            )

            return construct_final_prompt(
                concise_instruction,
                "No supporting context was retrieved.",
                query,
            )
