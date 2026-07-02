# ============================================================
# llm/ragent_client.py
# LLM Client Wrapper (Groq API Blocking)
# ============================================================
from __future__ import annotations
import os
from typing import Any
from groq import Groq
from utils.observability import ProfileBlock, MetricsRegistry

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
    **kwargs: Any,
) -> str:
    if not prompt:
        return ""
    MetricsRegistry.get().observe("llm_prompt_chars", len(prompt))
    client = _get_groq_client()
    accumulated = []
    
    with ProfileBlock("LLMGeneration"):
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                accumulated.append(chunk.choices[0].delta.content)
                
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
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **kwargs,
        )
        response = completion.choices[0].message.content or ""

    MetricsRegistry.get().observe("llm_decision_output_chars", len(response))
    return response
