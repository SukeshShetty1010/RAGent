# ============================================================
# llm/ragent_client.py
# LLM Client Wrapper (FULLY OBSERVABLE, PROD-SAFE)
# ============================================================

from __future__ import annotations

import modal
from typing import Any

from utils.observability import ProfileBlock, MetricsRegistry


# ------------------------------------------------------------
# Lazy Modal Cls Binding
# ------------------------------------------------------------

# Modal deployment declared in llm/modal_llm.py:
#   app = modal.App("gemma-3-12b-it-vllm")
_MODAL_APP_NAME = "gemma-3-12b-it-vllm"
_MODAL_CLASS_TAG = "Gemma312BVLLM"

_remote_instance = None


def _get_remote_llm():
    """
    Lazily resolve and cache a bound instance of Gemma312BVLLM.

    WHY THIS EXISTS:
    - Ensures MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
      are loaded BEFORE Modal client initialization
    - Prevents import-time auth failures
    - Works in Docker / fresh machines / CI
    """
    global _remote_instance

    if _remote_instance is None:
        cls = modal.Cls.from_name(
            _MODAL_APP_NAME,
            _MODAL_CLASS_TAG,
        )
        _remote_instance = cls()

    return _remote_instance


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
        accumulated = []
        for chunk in llm.generate.remote_gen(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            text = (
                chunk.decode()
                if isinstance(chunk, (bytes, bytearray))
                else str(chunk)
            )
            accumulated.append(text)

    response = "".join(accumulated)

    # --------------------------------------------------------
    # Post-call metrics (best-effort)
    # --------------------------------------------------------

    if isinstance(response, str):
        MetricsRegistry.get().observe(
            "llm_output_chars",
            len(response),
        )

    return response
