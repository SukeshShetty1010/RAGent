# ============================================================
# llm/ragent_client.py
# LLM Client Wrapper (FULLY OBSERVABLE)
# ============================================================

from __future__ import annotations

import modal
from typing import Any, Dict

from tests.observability import ProfileBlock, MetricsRegistry


# ------------------------------------------------------------
# Underlying Modal Remote Function (UNCHANGED)
# ------------------------------------------------------------

_raw_chat_completion = modal.Function.from_name(
    "rag-llama3-3b",
    "chat_completion_remote",
)


# ------------------------------------------------------------
# Observable Wrapper
# ------------------------------------------------------------

def chat_completion_remote(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    **kwargs: Any,
) -> str:
    """
    Observable wrapper around Modal LLM invocation.

    Measures:
    - LLM generation latency
    - Prompt size (chars)
    - Output size (chars)

    NOTE:
    Token counts are approximated unless returned by the model.
    """

    MetricsRegistry.get().observe(
        "llm_prompt_chars", len(prompt)
    )

    with ProfileBlock("LLMGeneration"):
        response = _raw_chat_completion.remote(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    # --------------------------------------------------------
    # Post-call metrics (best-effort)
    # --------------------------------------------------------

    if isinstance(response, str):
        MetricsRegistry.get().observe(
            "llm_output_chars", len(response)
        )

    return response
