# ============================================================
# llm/ragent_client.py
# LLM Client Wrapper (FULLY OBSERVABLE, PROD-SAFE)
# ============================================================

from __future__ import annotations

import modal
from typing import Any

from tests.observability import ProfileBlock, MetricsRegistry


# ------------------------------------------------------------
# Lazy Modal Function Binding (CRITICAL FIX)
# ------------------------------------------------------------

_raw_chat_completion = None


def _get_remote_llm():
    """
    Lazily resolve the Modal remote function.

    WHY THIS EXISTS:
    - Ensures MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
      are loaded BEFORE Modal client initialization
    - Prevents import-time auth failures
    - Works in Docker / fresh machines / CI
    """
    global _raw_chat_completion

    if _raw_chat_completion is None:
        _raw_chat_completion = modal.Function.from_name(
            "rag-smollm3-3b",
            "chat_completion_remote",
        )

    return _raw_chat_completion


# ------------------------------------------------------------
# Observable Wrapper (PUBLIC API)
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
    - Token counts are approximated unless returned by the model
    - All auth handled via environment (server-side only)
    """

    if not prompt:
        return ""

    MetricsRegistry.get().observe(
        "llm_prompt_chars",
        len(prompt),
    )

    llm = _get_remote_llm()

    with ProfileBlock("LLMGeneration"):
        response = llm.remote(
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
            "llm_output_chars",
            len(response),
        )

    return response
