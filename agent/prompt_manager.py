# ============================================================
# agent/prompt_manager.py
# Capability-Aware Prompt Budgeting Engine
# ============================================================

from __future__ import annotations

import logging
from typing import List, Dict, Any

from agent.task_router import TaskType
from agent.capability.capability_types import AnswerCapability
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

    Responsibilities:
    - Enforce hard prompt budget
    - Select task-specific instruction templates
    - Enforce honesty constraints based on AnswerCapability
    - NEVER decide feasibility (CapabilityAssessor owns that)
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
            # Capability guardrails
            # ------------------------------------------------
            if capability == AnswerCapability.INSUFFICIENT:
                MetricsRegistry.get().record(
                    "prompt_mode", "insufficient"
                )
                return self._insufficient_prompt(query)

            # ------------------------------------------------
            # Prepare context (highest value payload)
            # ------------------------------------------------
            with ProfileBlock("ContextFormatting"):
                context_chunks = list(chunks)
                context_block = self._format_context(context_chunks)

            # ------------------------------------------------
            # Select instructions (task + capability)
            # ------------------------------------------------
            verbose_instruction = self._get_verbose_instruction(
                task, capability
            )
            concise_instruction = self._get_concise_instruction(
                task, capability
            )

            # =================================================
            # ATTEMPT 1 — VERBOSE + FULL CONTEXT
            # =================================================
            prompt = self._construct_final_string(
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
            prompt = self._construct_final_string(
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
                context_block = self._format_context(context_chunks)

                prompt = self._construct_final_string(
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
            # ABSOLUTE FAIL-SAFE
            # ------------------------------------------------
            MetricsRegistry.get().record(
                "prompt_budget_mode", "minimal"
            )

            return self._construct_final_string(
                concise_instruction,
                "No supporting context was retrieved.",
                query,
            )

    # ========================================================
    # Core Assembly Helper
    # ========================================================

    @staticmethod
    def _construct_final_string(
        instruction: str,
        context_block: str,
        query: str,
    ) -> str:
        return (
            f"{instruction}\n\n"
            f"=== BEGIN CONTEXT ===\n"
            f"{context_block}\n"
            f"=== END CONTEXT ===\n\n"
            f"=== USER QUERY ===\n"
            f"{query}\n\n"
            f"=== ANSWER ===\n"
        )

    # ========================================================
    # Context Formatting
    # ========================================================

    def _format_context(
        self,
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

    # ========================================================
    # Instruction Dispatch (Task + Capability)
    # ========================================================

    def _get_verbose_instruction(
        self,
        task: TaskType,
        capability: AnswerCapability,
    ) -> str:
        if task == TaskType.COMPARISON:
            return self._comparison_verbose(capability)
        if task == TaskType.LISTICLE:
            return self._listicle_verbose(capability)
        if task == TaskType.FACTUAL:
            return self._factual_verbose(capability)
        return self._open_verbose(capability)

    def _get_concise_instruction(
        self,
        task: TaskType,
        capability: AnswerCapability,
    ) -> str:
        if task == TaskType.COMPARISON:
            return self._comparison_concise(capability)
        if task == TaskType.LISTICLE:
            return self._listicle_concise(capability)
        if task == TaskType.FACTUAL:
            return self._factual_concise(capability)
        return self._open_concise(capability)

    # ========================================================
    # Instruction Templates
    # ========================================================

    # ------------------------
    # COMPARISON
    # ------------------------

    @staticmethod
    def _comparison_verbose(capability: AnswerCapability) -> str:
        base = (
            "Compare entities across these dimensions:\n"
            "1. Gameplay\n"
            "2. Story\n"
            "3. World Design\n"
            "4. Tone\n"
            "5. Systems\n\n"
            "Use only the provided context.\n"
            "Cite sources for all factual statements.\n"
        )

        if capability == AnswerCapability.PARTIAL:
            base += (
                "\nIMPORTANT:\n"
                "If the context lacks sufficient information for any entity "
                "or dimension, explicitly state what is missing instead of "
                "guessing.\n"
            )

        return base

    @staticmethod
    def _comparison_concise(capability: AnswerCapability) -> str:
        base = (
            "Compare entities by Gameplay, Story, World Design, "
            "Tone, and Systems using only the context. Cite sources."
        )

        if capability == AnswerCapability.PARTIAL:
            base += (
                " Explicitly note missing information where applicable."
            )

        return base

    # ------------------------
    # LISTICLE
    # ------------------------

    @staticmethod
    def _listicle_verbose(capability: AnswerCapability) -> str:
        base = (
            "Produce an ordered list strictly from the provided context.\n"
            "Preserve original ordering when present.\n"
            "Do not invent items.\n"
            "Cite sources for each item.\n"
        )

        if capability == AnswerCapability.PARTIAL:
            base += (
                "\nIMPORTANT:\n"
                "If the list is incomplete, clearly state that the list "
                "represents a partial result based on available context.\n"
            )

        return base

    @staticmethod
    def _listicle_concise(capability: AnswerCapability) -> str:
        base = "Create an ordered list from context only. Cite sources."

        if capability == AnswerCapability.PARTIAL:
            base += " Note that the list may be incomplete."

        return base

    # ------------------------
    # FACTUAL
    # ------------------------

    @staticmethod
    def _factual_verbose(capability: AnswerCapability) -> str:
        base = (
            "Answer concisely using only the provided context.\n"
            "Limit the response to factual information.\n"
            "Cite all facts.\n"
        )

        if capability == AnswerCapability.PARTIAL:
            base += (
                "\nIMPORTANT:\n"
                "If the context does not fully answer the question, "
                "state what information is missing instead of guessing.\n"
            )

        return base

    @staticmethod
    def _factual_concise(capability: AnswerCapability) -> str:
        base = "Answer factually from the context. Cite sources."

        if capability == AnswerCapability.PARTIAL:
            base += " Note missing information explicitly."

        return base

    # ------------------------
    # OPEN
    # ------------------------

    @staticmethod
    def _open_verbose(capability: AnswerCapability) -> str:
        base = (
            "Answer using only the provided context.\n"
            "Organize clearly by topic.\n"
            "Cite all factual claims.\n"
        )

        if capability == AnswerCapability.PARTIAL:
            base += (
                "\nIMPORTANT:\n"
                "If parts of the answer cannot be supported by the context, "
                "clearly state those limitations.\n"
            )

        return base

    @staticmethod
    def _open_concise(capability: AnswerCapability) -> str:
        base = "Answer using the context. Cite sources."

        if capability == AnswerCapability.PARTIAL:
            base += " State any limitations clearly."

        return base

    # ========================================================
    # INSUFFICIENT PROMPT (NO LLM HALLUCINATION)
    # ========================================================

    @staticmethod
    def _insufficient_prompt(query: str) -> str:
        """
        Used when AnswerCapability == INSUFFICIENT.
        LLM should NOT be asked to infer or guess.
        """
        return (
            "The system does not have sufficient reliable information "
            "to answer the following request safely.\n\n"
            f"USER QUERY:\n{query}\n\n"
            "Respond with a brief, honest refusal explaining that "
            "the available evidence is insufficient."
        )
