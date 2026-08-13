# ============================================================
# llm/ragent_client_streaming.py
# Streaming LLM Client Wrapper (Groq API)
# ============================================================
from __future__ import annotations
import logging
import os
from typing import Any, Generator, Optional, Callable
import groq
from groq import Groq
from utils.observability import ProfileBlock, MetricsRegistry
from llm.ragent_client import (
    _get_groq_client,
    _get_modal_llm,
    _record_usage,
    _GROQ_MODEL,
)

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
    client = _get_groq_client()
    accumulated = []

    with ProfileBlock("LLMGenerationStreaming"):
        try:
            stream = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                # See ragent_client.py's chat_completion_remote for why
                # extra_body is used instead of a typed stream_options kwarg.
                extra_body={"stream_options": {"include_usage": True}},
                **kwargs,
            )

            for chunk in stream:
                # Final usage-only chunk has an empty choices list — must not
                # be yielded into the SSE stream, and [0] on it would raise.
                if not chunk.choices:
                    usage = chunk.usage or (chunk.x_groq.usage if chunk.x_groq else None)
                    _record_usage(usage)
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    accumulated.append(text)
                    if on_chunk:
                        on_chunk(text)
                    yield text
        except groq.RateLimitError as exc:
            # Groq validates the request and returns 429 before any chunk
            # is streamed, so this fires before anything has been yielded
            # — no risk of yielding a duplicate prefix from both models.
            logger.warning(f"Groq rate-limited, falling back to Modal Gemma: {exc}")
            MetricsRegistry.get().inc("llm_fallback_modal")
            with ProfileBlock("ModalLLMFallback"):
                llm = _get_modal_llm()
                for text in llm.generate.remote_gen(
                    prompt, max_new_tokens=max_tokens, temperature=temperature
                ):
                    accumulated.append(text)
                    if on_chunk:
                        on_chunk(text)
                    yield text

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
