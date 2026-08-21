"""
evaluation/gemini_judge_llm.py

Gemini RAGAS judge backend. Replaces the old Modal-hosted Gemma judge
(evaluation/modal_judge_llm.py, deleted) — that one needed a custom
BaseRagasLLM implementation because Modal's generate() isn't
OpenAI-compatible. Gemini is, via its OpenAI-compat endpoint (see
llm/gemini_client.py), so this is just langchain-openai's ChatOpenAI
pointed at Gemini's base URL, wrapped the same way the Groq judge
already is in evaluation/ragas_eval.py.

Like Groq (see the ChatGroq judge in evaluation/ragas_eval.py), Gemini's
OpenAI-compat endpoint rejects n>1 ("Multiple candidates is not enabled
for this model"), so ragas's self-consistency sampling (answer_relevancy's
question generation, faithfulness) must bypass it the same way.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from llm.gemini_client import GEMINI_MODEL, _GEMINI_BASE_URL

JUDGE_MODEL_ID = f"gemini:{GEMINI_MODEL}"


def build_gemini_judge() -> LangchainLLMWrapper:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=GEMINI_MODEL,
            base_url=_GEMINI_BASE_URL,
            api_key=api_key,
            temperature=0.0,
            max_tokens=8192,
        ),
        bypass_n=True,
    )
