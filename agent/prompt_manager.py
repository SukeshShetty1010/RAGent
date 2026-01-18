# ============================================================
# agent/prompt_manager.py
# Dynamic Prompt Budgeting Engine + Slim Comparison Template
# (FULLY OBSERVABLE)
# ============================================================

from __future__ import annotations

import logging
from typing import List, Dict, Any

from agent.task_router import TaskType
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
    Prompt container responsible for enforcing a HARD total
    character budget across:

      1. Instructions (boilerplate)
      2. Evidence (context)
      3. Query (mandatory)

    Hierarchy of importance:
      Query > Evidence > Structure > Instructions
    """

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def generate_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        task: TaskType,
    ) -> str:
        """
        Generate a prompt that NEVER exceeds MAX_TOTAL_PROMPT_CHARS.

        Degradation strategy:
          Attempt 1: Verbose instructions + full context
          Fallback A: Concise instructions + full context
          Fallback B: Concise instructions + truncated context (tail-drop)
        """

        with ProfileBlock("PromptConstruction"):

            # ------------------------------------------------
            # Prepare context (highest value payload)
            # ------------------------------------------------
            with ProfileBlock("ContextFormatting"):
                context_chunks = list(chunks)  # copy; may truncate
                context_block = self._format_context(context_chunks)

            # ------------------------------------------------
            # Select instruction sets
            # ------------------------------------------------
            verbose_instruction = self._get_verbose_instruction(task)
            concise_instruction = self._get_concise_instruction(task)

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
            # Drop lowest-priority chunks (from the end) until it fits
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
            logger.warning(
                "⚠️ Prompt severely constrained — context removed entirely"
            )

            MetricsRegistry.get().record(
                "prompt_budget_mode", "minimal"
            )

            return self._construct_final_string(
                concise_instruction,
                "No supporting context was retrieved.",
                query,
            )

    # ========================================================
    # Core Assembly Helper (DRY)
    # ========================================================

    @staticmethod
    def _construct_final_string(
        instruction: str,
        context_block: str,
        query: str,
    ) -> str:
        """
        Assemble the FINAL serialized prompt string.

        This method is the single source of truth for
        budget measurement via len(prompt).
        """
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
    # Instruction Dispatch
    # ========================================================

    def _get_verbose_instruction(self, task: TaskType) -> str:
        if task == TaskType.COMPARISON:
            return self._comparison_instruction_verbose()
        if task == TaskType.LISTICLE:
            return self._listicle_instruction_verbose()
        if task == TaskType.FACTUAL:
            return self._factual_instruction_verbose()
        return self._open_instruction_verbose()

    def _get_concise_instruction(self, task: TaskType) -> str:
        if task == TaskType.COMPARISON:
            return self._comparison_instruction_concise()
        if task == TaskType.LISTICLE:
            return self._listicle_instruction_concise()
        if task == TaskType.FACTUAL:
            return self._factual_instruction_concise()
        return self._open_instruction_concise()

    # ========================================================
    # VERBOSE INSTRUCTIONS (SLIM, HIGH-SIGNAL)
    # ========================================================

    @staticmethod
    def _comparison_instruction_verbose() -> str:
        return (
            "Compare entities across these dimensions:\n"
            "1. Gameplay (core mechanics, loops, player agency)\n"
            "2. Story (narrative focus, themes, delivery)\n"
            "3. World Design (structure, exploration, activities)\n"
            "4. Tone (emotional register, satire vs seriousness)\n"
            "5. Systems (progression, AI, combat, economy)\n\n"
            "Use only the provided context. "
            "Cite sources at the end of each factual statement."
        )

    @staticmethod
    def _listicle_instruction_verbose() -> str:
        return (
            "Produce an ordered list strictly from the provided context.\n"
            "Preserve original ordering when present.\n"
            "Do not invent items.\n"
            "Cite sources for each item."
        )

    @staticmethod
    def _factual_instruction_verbose() -> str:
        return (
            "Answer concisely using only the provided context.\n"
            "Limit to factual information.\n"
            "Cite all facts."
        )

    @staticmethod
    def _open_instruction_verbose() -> str:
        return (
            "Answer using only the provided context.\n"
            "Organize clearly by topic.\n"
            "Cite all factual claims."
        )

    # ========================================================
    # CONCISE INSTRUCTIONS (ULTRA-DENSE)
    # ========================================================

    @staticmethod
    def _comparison_instruction_concise() -> str:
        return (
            "Compare entities by: Gameplay, Story, World Design, "
            "Tone, Systems. Use context only. Cite sources."
        )

    @staticmethod
    def _listicle_instruction_concise() -> str:
        return (
            "Create an ordered list from context only. Cite sources."
        )

    @staticmethod
    def _factual_instruction_concise() -> str:
        return (
            "Answer factually from the context. Cite sources."
        )

    @staticmethod
    def _open_instruction_concise() -> str:
        return (
            "Answer using the context. Cite sources."
        )
