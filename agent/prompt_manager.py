# ============================================================
# agent/prompt_manager.py
# Step 7: Task-Specific Prompt Templates
# ============================================================

from __future__ import annotations

from typing import List, Dict, Any

from agent.task_router import TaskType


# ============================================================
# Prompt Manager
# ============================================================

class PromptManager:
    """
    Instruction Layer.

    Translates:
    - TaskType
    - Assembled Context
    - User Query

    into a final, structured prompt string for the LLM.

    This module:
    - Obeys TaskType strictly
    - Performs NO reasoning
    - Uses deterministic templates only
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
        Generate the final prompt string for the LLM.
        """

        context_block = self._format_context(chunks)

        if task == TaskType.COMPARISON:
            system_instruction = self._comparison_instruction()
        elif task == TaskType.LISTICLE:
            system_instruction = self._listicle_instruction()
        elif task == TaskType.FACTUAL:
            system_instruction = self._factual_instruction()
        else:
            system_instruction = self._open_instruction()

        prompt = (
            f"{system_instruction}\n\n"
            f"=== BEGIN CONTEXT ===\n"
            f"{context_block}\n"
            f"=== END CONTEXT ===\n\n"
            f"=== USER QUERY ===\n"
            f"{query}\n\n"
            f"=== ANSWER ===\n"
        )

        return prompt

    # ========================================================
    # Context Formatting
    # ========================================================

    def _format_context(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Convert chunks into a single formatted context string.
        """

        if not chunks:
            return "No supporting context was retrieved."

        formatted_chunks: List[str] = []

        for c in chunks:
            source_title = c.get("source_title") or "Unknown Source"
            source_type = c.get("source_type") or "local"
            content = c.get("content") or ""

            formatted_chunks.append(
                f"[Source: {source_title} | Type: {source_type}]\n"
                f"{content}\n"
            )

        return "\n".join(formatted_chunks)

    # ========================================================
    # System Instructions (Templates)
    # ========================================================

    @staticmethod
    def _comparison_instruction() -> str:
        return (
            "You are an objective video game analyst. "
            "Your task is to compare two or more entities strictly based on the provided context.\n\n"
            "You must output your answer in this exact structure:\n"
            "1. **Overview**: High-level summary of the comparison.\n"
            "2. **Gameplay**: Contrast mechanics, difficulty, and loops.\n"
            "3. **Story/Atmosphere**: Contrast narrative and world design.\n"
            "4. **Conclusion**: A final summary statement.\n\n"
            "Constraint:\n"
            "If the context is missing information for one side, explicitly state:\n"
            "\"I lack sufficient information to compare [Topic].\"\n\n"
            "Global Citation Rule:\n"
            "You must cite your sources. When using information from a specific chunk, "
            "append (Source: [Source Title]) at the end of the sentence."
        )

    @staticmethod
    def _listicle_instruction() -> str:
        return (
            "You are a helpful gaming guide editor. "
            "Your task is to create an ordered list based on the provided context.\n\n"
            "You must preserve the original order of the items found in the context if they are numbered.\n"
            "Format your output as:\n"
            "1. **[Item Name]**: [Description]\n"
            "2. **[Item Name]**: [Description]\n"
            "...\n\n"
            "Constraint:\n"
            "Do not invent list items. Only include items present in the context.\n\n"
            "Global Citation Rule:\n"
            "You must cite your sources. When using information from a specific chunk, "
            "append (Source: [Source Title]) at the end of the sentence."
        )

    @staticmethod
    def _factual_instruction() -> str:
        return (
            "You are a precise gaming database assistant. "
            "Answer the user's question concisely.\n\n"
            "Structure Instruction:\n"
            "Provide a direct, factual answer in 1–2 paragraphs. Do not use filler words.\n\n"
            "Global Citation Rule:\n"
            "You must cite your sources. When using information from a specific chunk, "
            "append (Source: [Source Title]) at the end of the sentence."
        )

    @staticmethod
    def _open_instruction() -> str:
        return (
            "You are a knowledgeable gaming assistant. "
            "Provide a comprehensive answer based on the context.\n\n"
            "Structure Instruction:\n"
            "Use clear paragraphs and headings where appropriate.\n\n"
            "Global Citation Rule:\n"
            "You must cite your sources. When using information from a specific chunk, "
            "append (Source: [Source Title]) at the end of the sentence."
        )
