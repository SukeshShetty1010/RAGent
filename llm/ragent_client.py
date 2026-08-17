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

# llama-3.1-8b-instant was retired: Groq answers it with 404
# "does not exist or you do not have access to it" (confirmed live
# 2026-08-17 against the account's own /models listing, which no longer
# offers any llama chat model). That silently killed the entire fallback
# — every Gemini failure produced a 404 instead of an answer, which only
# went unnoticed because Gemini rarely failed. Env-overridable for the
# same reason GEMINI_MODEL is: these get retired without notice, and the
# fix should not need a code change.
# openai/gpt-oss-120b chosen over gpt-oss-20b (both live-tested, both
# ~0.7s) for answer quality, and over qwen3.6-27b, which emits raw
# <think> blocks into message content and would put reasoning text in
# front of users.
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Output cap for user-facing answer generation. 512 was too low and cut
# real answers mid-sentence: the "Far Cry 5" prompt (4,243 chars, 861
# prompt tokens) needs 692 completion tokens and returned
# finish_reason="length" at 512 — the truncation reached the UI as a
# sentence ending in "a doomsday cult led". Measured 2026-08-17; 2048
# returns finish_reason="stop" with headroom for longer comparison and
# listicle answers. Shared by the blocking and streaming clients so the
# Gemini path and its Groq fallback cannot disagree about answer length.
# chat_completion_decision keeps its own much smaller cap: it emits
# bounded JSON, where a large cap would only widen a runaway.
_ANSWER_MAX_TOKENS = 2048


def _chunk_usage(chunk: Any) -> Any:
    """Pull the usage object off a streaming chunk, whichever way it arrives.

    Providers disagree about where usage lands under
    stream_options.include_usage, and reading only one shape silently
    loses the token counts:
      - Groq sends a final chunk with an EMPTY choices list, carrying
        usage either directly or under x_groq.
      - Gemini's OpenAI-compat surface sends no choice-less chunk at all;
        usage rides on the SAME final chunk that carries finish_reason
        (confirmed live 2026-08-17). The old choice-less-only check never
        matched, so every Gemini stream recorded zero tokens and the
        prompt_tokens/completion_tokens/cost_usd KPIs were always null.
    Returns None when the chunk carries no usage, which is most of them.
    """
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        return usage
    x_groq = getattr(chunk, "x_groq", None)
    if x_groq is not None:
        return getattr(x_groq, "usage", None)
    return None


def _record_finish_reason(reason: Any) -> None:
    """Record why generation stopped, so a truncated answer is visible.

    "length" means the model hit max_tokens and the answer is cut off
    mid-sentence. Discarding this made truncation indistinguishable from
    a complete short answer anywhere downstream — the engine now lifts it
    into the KPIs so the UI can say the answer is incomplete instead of
    presenting a severed sentence as finished.
    """
    if not reason:
        return
    MetricsRegistry.get().record("llm_finish_reason", str(reason))


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
        # Same defence as _get_gemini_api_key(): python-dotenv strips
        # quotes from a quoted value, but `docker run --env-file` and
        # similar raw injection pass the quotes through, turning every
        # request into a 401.
        api_key = api_key.strip().strip('"').strip("'")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def chat_completion_remote(
    prompt: str,
    max_tokens: int = _ANSWER_MAX_TOKENS,
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
            _record_finish_reason(completion.choices[0].finish_reason)
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
                        groq_usage = _chunk_usage(chunk) or groq_usage
                        # The final usage-only chunk (stream_options.include_usage) has
                        # an empty choices list — indexing [0] on it raises IndexError.
                        if not chunk.choices:
                            continue
                        _record_finish_reason(chunk.choices[0].finish_reason)
                        if chunk.choices[0].delta.content:
                            accumulated.append(chunk.choices[0].delta.content)
                    _record_usage(groq_usage, _GROQ_MODEL)
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
    max_tokens: int = 320,
    temperature: float = 0.0,
    **kwargs: Any,
) -> str:
    """
    Non-streaming, single-shot LLM call for bounded structured decisions
    (JSON-object responses). Distinct from chat_completion_remote, which
    is streaming and intended for user-facing answer generation.

    Still deliberately bounded, but no longer 150: the Groq fallback is
    now a reasoning model whose hidden reasoning tokens are billed
    against max_tokens, and Groq rejects reasoning_effort="none" (only
    low/medium/high). A decision measured 88 reasoning tokens before
    emitting 44 characters of JSON, so 150 left almost no margin — and
    running out returns empty content rather than an error, which
    degrades WebSearchDecision to its deterministic fallback silently.
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
