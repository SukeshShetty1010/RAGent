# ============================================================
# llm/ragent_client.py
# LLM Client Wrapper — Gemini primary, Groq fallback (blocking)
# ============================================================
from __future__ import annotations
import logging
import os
from typing import Any
import groq
from groq import Groq
from utils.observability import ProfileBlock, MetricsRegistry
from utils.usage_counter import UsageCounter
from llm.pricing import estimate_cost
from llm.gemini_client import _get_gemini_client, GEMINI_MODEL

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.1-8b-instant"


def _record_usage(usage: Any, model: str) -> None:
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
        estimate_cost(model, prompt_tokens, completion_tokens),
    )


def last_used_model() -> str:
    """Which model actually served the most recent generation call, per the
    llm_provider_* counters incremented by chat_completion_remote/decision/
    streaming. Used to report the real model to tracing instead of a
    hardcoded string."""
    counters = MetricsRegistry.get().generate_report()["counters"]
    if counters.get("llm_provider_gemini", 0) > 0:
        return GEMINI_MODEL
    if counters.get("llm_provider_groq", 0) > 0:
        return _GROQ_MODEL
    return "unknown"

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


def chat_completion_remote(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    surface: str = "chat",
    **kwargs: Any,
) -> str:
    if not prompt:
        return ""
    MetricsRegistry.get().observe("llm_prompt_chars", len(prompt))
    accumulated = []

    with ProfileBlock("LLMGeneration"):
        try:
            client = _get_gemini_client()
            completion = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                reasoning_effort="none",
                **kwargs,
            )
            accumulated = [completion.choices[0].message.content or ""]
            _record_usage(completion.usage, GEMINI_MODEL)
            MetricsRegistry.get().inc("llm_provider_gemini")
            usage = completion.usage
            UsageCounter.get().record(
                "gemini", surface,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        except Exception as exc:
            logger.warning(f"Gemini unavailable, falling back to Groq: {exc}")
            MetricsRegistry.get().inc("llm_provider_groq_fallback")
            with ProfileBlock("GroqFallback"):
                client = _get_groq_client()
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
                    accumulated = []
                    groq_usage = None
                    for chunk in stream:
                        # The final usage-only chunk (stream_options.include_usage) has
                        # an empty choices list — indexing [0] on it raises IndexError.
                        if not chunk.choices:
                            groq_usage = chunk.usage or (chunk.x_groq.usage if chunk.x_groq else None)
                            _record_usage(groq_usage, _GROQ_MODEL)
                            continue
                        if chunk.choices[0].delta.content:
                            accumulated.append(chunk.choices[0].delta.content)
                    MetricsRegistry.get().inc("llm_provider_groq")
                    UsageCounter.get().record(
                        "groq", surface,
                        prompt_tokens=getattr(groq_usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(groq_usage, "completion_tokens", 0) or 0,
                        fallback=True,
                    )
                except groq.RateLimitError as groq_exc:
                    logger.warning(f"Groq also rate-limited: {groq_exc}")
                    accumulated = []

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

    with ProfileBlock("LLMDecision"):
        try:
            client = _get_gemini_client()
            completion = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                reasoning_effort="none",
                **kwargs,
            )
            response = completion.choices[0].message.content or ""
            _record_usage(completion.usage, GEMINI_MODEL)
            MetricsRegistry.get().inc("llm_provider_gemini")
            UsageCounter.get().record(
                "gemini", "decision",
                prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
            )
        except Exception as exc:
            logger.warning(f"Gemini unavailable, falling back to Groq: {exc}")
            MetricsRegistry.get().inc("llm_provider_groq_fallback")
            client = _get_groq_client()
            completion = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                **kwargs,
            )
            response = completion.choices[0].message.content or ""
            _record_usage(completion.usage, _GROQ_MODEL)
            MetricsRegistry.get().inc("llm_provider_groq")
            UsageCounter.get().record(
                "groq", "decision",
                prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
                fallback=True,
            )

    MetricsRegistry.get().observe("llm_decision_output_chars", len(response))
    return response
