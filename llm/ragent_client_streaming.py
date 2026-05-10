# ============================================================
# llm/ragent_client_streaming.py
# Streaming LLM Client Wrapper (GENERATOR-BASED)
# ============================================================
"""
Streaming-capable LLM client for conversational UI.

Targets the Modal-hosted Gemma 3 12B Instruct vLLM deployment:
  - App:      gemma-3-12b-it-vllm               (modal_llm.py)
  - Class:    Gemma312BVLLM
  - Method:   generate  (L40S GPU, Modal generator)

The remote ``generate`` method is a Modal generator function decorated
with ``@modal.method()`` on a ``@app.cls``-decorated class.  The correct
way to call a method on a Modal Cls from another process is:

    Gemma312BVLLM = modal.Cls.from_name(APP_NAME, CLASS_TAG)
    instance      = Gemma312BVLLM()
    for chunk in instance.generate.remote_gen(prompt, ...):
        ...

This module provides both blocking and streaming interfaces.

Usage:
    # Blocking mode
    response = chat_completion_remote(prompt)

    # Streaming mode (generator)
    for chunk in chat_completion_streaming(prompt):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import modal
from typing import Any, Generator, Optional, Callable

from utils.observability import ProfileBlock, MetricsRegistry


# ------------------------------------------------------------
# Deployment constants
# ------------------------------------------------------------

# The Modal App name as declared in modal_llm.py:
#   app = modal.App("gemma-3-12b-it-vllm")
_MODAL_APP_NAME = "gemma-3-12b-it-vllm"

# The @app.cls class name hosting the generate method.
_MODAL_CLASS_TAG = "Gemma312BVLLM"

# The @modal.method name on Gemma312BVLLM that streams tokens.
_MODAL_METHOD_TAG = "generate"


# ------------------------------------------------------------
# Lazy Modal Cls Binding
# ------------------------------------------------------------

_remote_cls: Optional[Any] = None          # modal.Cls proxy
_remote_instance: Optional[Any] = None    # bound instance of Gemma312BVLLM


def _get_remote_instance() -> Any:
    """
    Lazily resolve and cache a bound instance of Gemma312BVLLM from the
    gemma-3-12b-it-vllm deployment.

    modal.Cls.from_name is lazy: it defers hydration until the first
    actual call, so no network round-trip occurs here.

    Returns a bound Cls instance whose methods (e.g. .generate) expose
    .remote(), .remote_gen(), and .spawn() just like modal.Function does
    for standalone functions.
    """
    global _remote_cls, _remote_instance

    if _remote_instance is None:
        _remote_cls = modal.Cls.from_name(
            _MODAL_APP_NAME,
            _MODAL_CLASS_TAG,
        )
        # Calling the Cls proxy returns a bound instance.
        # This does NOT spin up a container; it is a purely local object.
        _remote_instance = _remote_cls()

    return _remote_instance


# ------------------------------------------------------------
# Blocking API (Original interface — preserved)
# ------------------------------------------------------------

def chat_completion_remote(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    **kwargs: Any,
) -> str:
    """
    Blocking wrapper around Modal Gemma312BVLLM.generate invocation.

    Because ``generate`` is a generator method on the remote side,
    we consume it via remote_gen and join all chunks into a single
    string, preserving the blocking contract of the original API.

    Returns the full generated response as a string.
    """
    if not prompt:
        return ""

    MetricsRegistry.get().observe(
        "llm_prompt_chars",
        len(prompt),
    )

    instance = _get_remote_instance()
    accumulated: list[str] = []

    with ProfileBlock("LLMGeneration"):
        try:
            # generate is a Modal generator method; use remote_gen to
            # consume all yielded chunks and accumulate them.
            for chunk in instance.generate.remote_gen(
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
        except Exception:
            # Propagate — callers handle LLM errors via their own
            # try/except (e.g. execution_engine_streaming.py).
            raise

    response = "".join(accumulated)

    MetricsRegistry.get().observe(
        "llm_output_chars",
        len(response),
    )

    return response


# ------------------------------------------------------------
# Streaming API (Generator-based)
# ------------------------------------------------------------

def chat_completion_streaming(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    on_chunk: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Generator[str, None, str]:
    """
    Streaming generator for Modal Gemma312BVLLM.generate invocation.

    Uses Modal's remote_gen to stream token chunks from the
    Gemma 3 12B Instruct vLLM deployment in real time.

    Falls back to simulated streaming (word-by-word from a blocking
    call) if remote_gen is unavailable on the resolved method object.

    Yields:
        String chunks as they are generated

    Returns:
        The complete accumulated response

    Example:
        accumulated = ""
        for chunk in chat_completion_streaming(prompt):
            print(chunk, end="", flush=True)
            accumulated += chunk
    """
    if not prompt:
        return ""

    MetricsRegistry.get().observe(
        "llm_prompt_chars",
        len(prompt),
    )

    instance = _get_remote_instance()
    accumulated: list[str] = []

    with ProfileBlock("LLMGenerationStreaming"):
        # Resolve the bound method from the Cls instance.
        generate_method = instance.generate
        remote_gen = getattr(generate_method, "remote_gen", None)

        if callable(remote_gen):
            try:
                gen = remote_gen(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )

                # remote_gen returns a synchronous iterator over
                # the remote generator's yielded values.
                if hasattr(gen, "__aiter__"):
                    # Async generator path (unusual for modal.Cls method
                    # remote_gen, but handle defensively).
                    import asyncio

                    async def _consume_async():
                        async for chunk in gen:
                            text = (
                                chunk.decode()
                                if isinstance(chunk, (bytes, bytearray))
                                else str(chunk)
                            )
                            accumulated.append(text)
                            if on_chunk:
                                on_chunk(text)
                            yield text

                    loop = asyncio.new_event_loop()
                    try:
                        async_gen = _consume_async()
                        while True:
                            try:
                                chunk = loop.run_until_complete(
                                    async_gen.__anext__()
                                )
                                yield chunk
                            except StopAsyncIteration:
                                break
                    finally:
                        loop.close()

                else:
                    # Primary path: synchronous iterator from remote_gen.
                    for chunk in gen:
                        text = (
                            chunk.decode()
                            if isinstance(chunk, (bytes, bytearray))
                            else str(chunk)
                        )
                        accumulated.append(text)
                        if on_chunk:
                            on_chunk(text)
                        yield text

                full_response = "".join(accumulated)
                MetricsRegistry.get().observe(
                    "llm_output_chars", len(full_response)
                )
                return full_response

            except Exception:
                # Fall through to simulated-streaming fallback below.
                # Reset accumulator so fallback starts clean.
                accumulated.clear()

        # ----------------------------------------------------------
        # Fallback: blocking call + word-by-word simulated streaming.
        # This path is taken only when remote_gen is unavailable or
        # raised an unexpected error.
        # ----------------------------------------------------------
        response = generate_method.remote(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        # generate is a generator; if .remote() was called on it
        # Modal may return an iterable — drain it defensively.
        if hasattr(response, "__iter__") and not isinstance(response, str):
            parts: list[str] = []
            for item in response:
                text = (
                    item.decode()
                    if isinstance(item, (bytes, bytearray))
                    else str(item)
                )
                parts.append(text)
            response = "".join(parts)

        if isinstance(response, str):
            MetricsRegistry.get().observe(
                "llm_output_chars", len(response)
            )

            # Simulate streaming by yielding words.
            words = response.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                if on_chunk:
                    on_chunk(chunk)
                yield chunk

            return response

        return ""


# ------------------------------------------------------------
# Word-by-word streaming simulation
# ------------------------------------------------------------

def simulate_streaming(
    text: str,
    words_per_chunk: int = 3,
) -> Generator[str, None, None]:
    """
    Simulate streaming by yielding words in small chunks.

    This is useful for UI effects when true streaming
    is not available from the LLM backend.

    Args:
        text: The full text to stream
        words_per_chunk: Number of words to yield at a time

    Yields:
        String chunks of the specified size
    """
    words = text.split()

    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i : i + words_per_chunk]
        chunk = " ".join(chunk_words)

        # Add trailing space unless end of text
        if i + words_per_chunk < len(words):
            chunk += " "

        yield chunk
