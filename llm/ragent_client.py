# ============================================================
# llm/ragent_client.py
# LLM Client Wrapper (Groq API Blocking)
# ============================================================
from __future__ import annotations
import logging
import os
from typing import Any
import groq
from groq import Groq
from utils.observability import ProfileBlock, MetricsRegistry
from llm.pricing import estimate_cost

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.1-8b-instant"
_MODAL_LLM_APP = "gemma-3-12b-it-vllm"
_MODAL_LLM_CLASS = "Gemma312BVLLM"


def _record_usage(usage: Any) -> None:
    """Record prompt/completion tokens and estimated cost, if usage is present."""
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    registry = MetricsRegistry.get()
    registry.observe("llm_prompt_tokens", prompt_tokens)
    registry.observe("llm_completion_tokens", completion_tokens)
    registry.observe(
        "llm_cost_usd",
        estimate_cost(_GROQ_MODEL, prompt_tokens, completion_tokens),
    )

_groq_client = None

def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


_modal_llm = None

def _get_modal_llm() -> Any:
    """Lazy Modal client for the Gemma-3-12B fallback, matching the
    _get_embedder()/_get_reranker() lazy-binding pattern in
    retriever/rag_retriever.py — a module-level modal.Cls.from_name()
    call would make import fail without Modal credentials present."""
    global _modal_llm
    if _modal_llm is None:
        import modal
        Gemma312BVLLM = modal.Cls.from_name(_MODAL_LLM_APP, _MODAL_LLM_CLASS)
        _modal_llm = Gemma312BVLLM()
    return _modal_llm


def _generate_via_modal_fallback(
    prompt: str, max_tokens: int, temperature: float
) -> str:
    """Collect Gemma-3-12B's streamed chunks into one string.

    No token-usage/cost tracking here: Modal bills by GPU-second, not
    per-token, and generate() returns raw text with no usage payload —
    fabricating a token count would be worse than reporting none.
    """
    llm = _get_modal_llm()
    chunks = list(
        llm.generate.remote_gen(
            prompt, max_new_tokens=max_tokens, temperature=temperature
        )
    )
    return "".join(chunks)

def chat_completion_remote(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    **kwargs: Any,
) -> str:
    if not prompt:
        return ""
    MetricsRegistry.get().observe("llm_prompt_chars", len(prompt))
    client = _get_groq_client()
    accumulated = []
    
    with ProfileBlock("LLMGeneration"):
        try:
            stream = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                # Groq SDK 0.37.1 has no typed stream_options param; the
                # underlying API still honors it via extra_body (confirmed
                # live: final chunk arrives with empty choices + populated
                # usage only when this is set).
                extra_body={"stream_options": {"include_usage": True}},
                **kwargs,
            )
            for chunk in stream:
                # The final usage-only chunk (stream_options.include_usage) has
                # an empty choices list — indexing [0] on it raises IndexError.
                if not chunk.choices:
                    usage = chunk.usage or (chunk.x_groq.usage if chunk.x_groq else None)
                    _record_usage(usage)
                    continue
                if chunk.choices[0].delta.content:
                    accumulated.append(chunk.choices[0].delta.content)
        except groq.RateLimitError as exc:
            logger.warning(f"Groq rate-limited, falling back to Modal Gemma: {exc}")
            MetricsRegistry.get().inc("llm_fallback_modal")
            with ProfileBlock("ModalLLMFallback"):
                accumulated = [_generate_via_modal_fallback(prompt, max_tokens, temperature)]

    response = "".join(accumulated)
    MetricsRegistry.get().observe("llm_output_chars", len(response))
    return response

def chat_completion_decision(
    prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.0,
    **kwargs: Any,
) -> str:
    """
    Non-streaming, single-shot LLM call for bounded structured decisions
    (JSON-object responses). Distinct from chat_completion_remote, which
    is streaming and intended for user-facing answer generation.
    """
    if not prompt:
        return ""
    MetricsRegistry.get().observe("llm_decision_prompt_chars", len(prompt))
    client = _get_groq_client()

    with ProfileBlock("LLMDecision"):
        completion = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs,
        )
        response = completion.choices[0].message.content or ""
        _record_usage(completion.usage)

    MetricsRegistry.get().observe("llm_decision_output_chars", len(response))
    return response
