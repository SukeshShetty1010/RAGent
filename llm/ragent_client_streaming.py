# ============================================================
# llm/ragent_client_streaming.py
# Streaming LLM Client Wrapper (Groq API)
# ============================================================
from __future__ import annotations
import os
from typing import Any, Generator, Optional, Callable
from groq import Groq
from utils.observability import ProfileBlock, MetricsRegistry
from llm.ragent_client import _get_groq_client

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
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
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
