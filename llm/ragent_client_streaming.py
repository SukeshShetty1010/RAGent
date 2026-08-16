# ============================================================
# llm/ragent_client_streaming.py
# Streaming LLM Client Wrapper — Gemini primary, Groq fallback
# ============================================================
from __future__ import annotations
import logging
from typing import Any, Generator, Optional, Callable
import groq
from utils.observability import ProfileBlock, MetricsRegistry
from utils.usage_counter import UsageCounter
from llm.ragent_client import (
    _get_groq_client,
    _record_usage,
    _GROQ_MODEL,
)
from llm.gemini_client import _get_gemini_client, GEMINI_MODEL

logger = logging.getLogger(__name__)

def chat_completion_streaming(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    on_chunk: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Generator[str, None, str]:
    if not prompt:
        return ""

    MetricsRegistry.get().observe("llm_prompt_chars", len(prompt))
    accumulated = []
    any_yielded = False
    gemini_usage = None

    with ProfileBlock("LLMGenerationStreaming"):
        try:
            client = _get_gemini_client()
            stream = client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
                reasoning_effort="none",
                **kwargs,
            )
            for chunk in stream:
                if not chunk.choices:
                    gemini_usage = chunk.usage
                    _record_usage(gemini_usage, GEMINI_MODEL)
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    any_yielded = True
                    accumulated.append(text)
                    if on_chunk:
                        on_chunk(text)
                    yield text
            MetricsRegistry.get().inc("llm_provider_gemini")
            UsageCounter.get().record(
                "gemini", "chat",
                prompt_tokens=getattr(gemini_usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(gemini_usage, "completion_tokens", 0) or 0,
            )
        except Exception as exc:
            if any_yielded:
                # A chunk already reached the SSE stream — falling back now
                # would duplicate the prefix, so stop here instead.
                logger.error(f"Gemini failed mid-stream, aborting (no fallback): {exc}")
            else:
                # Nothing yielded yet, safe to retry the whole prompt on Groq.
                logger.warning(f"Gemini unavailable, falling back to Groq: {exc}")
                MetricsRegistry.get().inc("llm_provider_groq_fallback")
                with ProfileBlock("GroqFallback"):
                    try:
                        client = _get_groq_client()
                        groq_stream = client.chat.completions.create(
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
                        groq_usage = None
                        for chunk in groq_stream:
                            if not chunk.choices:
                                groq_usage = chunk.usage or (chunk.x_groq.usage if chunk.x_groq else None)
                                _record_usage(groq_usage, _GROQ_MODEL)
                                continue
                            text = chunk.choices[0].delta.content
                            if text:
                                accumulated.append(text)
                                if on_chunk:
                                    on_chunk(text)
                                yield text
                        MetricsRegistry.get().inc("llm_provider_groq")
                        UsageCounter.get().record(
                            "groq", "chat",
                            prompt_tokens=getattr(groq_usage, "prompt_tokens", 0) or 0,
                            completion_tokens=getattr(groq_usage, "completion_tokens", 0) or 0,
                            fallback=True,
                        )
                    except groq.RateLimitError as groq_exc:
                        # Groq validates the request and returns 429 before any
                        # chunk is streamed, so this fires before anything has
                        # been yielded from either provider — no duplicate prefix.
                        logger.error(f"Groq also rate-limited: {groq_exc}")

    full_response = "".join(accumulated)
    MetricsRegistry.get().observe("llm_output_chars", len(full_response))
    return full_response

def simulate_streaming(text: str, words_per_chunk: int = 3) -> Generator[str, None, None]:
    words = text.split()
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i : i + words_per_chunk]
        chunk = " ".join(chunk_words)
        if i + words_per_chunk < len(words):
            chunk += " "
        yield chunk
