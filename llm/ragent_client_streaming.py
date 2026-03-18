# ============================================================
# llm/ragent_client_streaming.py
# Streaming LLM Client Wrapper (GENERATOR-BASED)
# ============================================================
"""
Streaming-capable LLM client for conversational UI.

This module provides both blocking and streaming interfaces
to the Modal-hosted LLM service.

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

from tests.observability import ProfileBlock, MetricsRegistry


# ------------------------------------------------------------
# Lazy Modal Function Binding
# ------------------------------------------------------------

_raw_chat_completion = None


def _get_remote_llm():
    """
    Lazily resolve the Modal remote function.
    """
    global _raw_chat_completion

    if _raw_chat_completion is None:
        _raw_chat_completion = modal.Function.from_name(
            "rag-smollm3-3b",
            "chat_completion_remote",
        )

    return _raw_chat_completion


# ------------------------------------------------------------
# Blocking API (Original)
# ------------------------------------------------------------

def chat_completion_remote(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    **kwargs: Any,
) -> str:
    """
    Blocking wrapper around Modal LLM invocation.
    
    Returns the full generated response as a string.
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

    if isinstance(response, str):
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
    Streaming generator for Modal LLM invocation.
    
    Attempts to use Modal's streaming capabilities if available,
    otherwise falls back to simulated streaming from the full response.
    
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

    llm = _get_remote_llm()
    accumulated = []

    with ProfileBlock("LLMGenerationStreaming"):
        # Attempt to use streaming if available
        remote_gen = getattr(llm, "remote_gen", None)
        
        if callable(remote_gen):
            try:
                gen = remote_gen(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
                
                # Handle async generator
                if hasattr(gen, "__aiter__"):
                    import asyncio
                    
                    async def consume_async():
                        nonlocal accumulated
                        async for chunk in gen:
                            text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                            accumulated.append(text)
                            if on_chunk:
                                on_chunk(text)
                            yield text
                    
                    # Run in event loop
                    loop = asyncio.new_event_loop()
                    try:
                        async_gen = consume_async()
                        while True:
                            try:
                                chunk = loop.run_until_complete(async_gen.__anext__())
                                yield chunk
                            except StopAsyncIteration:
                                break
                    finally:
                        loop.close()
                else:
                    # Synchronous generator
                    for chunk in gen:
                        text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                        accumulated.append(text)
                        if on_chunk:
                            on_chunk(text)
                        yield text
                
                full_response = "".join(accumulated)
                MetricsRegistry.get().observe("llm_output_chars", len(full_response))
                return full_response
                
            except Exception:
                # Fall through to simulated streaming
                pass
        
        # Fallback: blocking call with simulated streaming
        response = llm.remote(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        if isinstance(response, str):
            MetricsRegistry.get().observe("llm_output_chars", len(response))
            
            # Simulate streaming by yielding words
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
        chunk_words = words[i:i + words_per_chunk]
        chunk = " ".join(chunk_words)
        
        # Add trailing space unless end of text
        if i + words_per_chunk < len(words):
            chunk += " "
        
        yield chunk
