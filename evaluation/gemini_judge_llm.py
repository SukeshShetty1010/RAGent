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

# Gemini's free tier caps *requests per minute* (measured live 2026-08-23:
# `limit: 15, model: gemini-3.5-flash-lite`), and a multi-hundred-call job
# like evaluation/ablation.py sits above that cap for its whole duration.
# ragas has no retry layer of its own -- ragas.executor catches the
# exception and records the sample as NaN -- so the OpenAI SDK's own
# retry budget is the only thing standing between an RPM throttle and a
# silently dropped sample. The default of 2 is not enough: a 429 under
# this cap asks for a ~45s wait, while 2 retries back off for ~1.5s
# total, so modes scored during a busy minute end up with a smaller `n`
# than modes scored during a quiet one -- an invisible bias in exactly
# the cross-mode comparison the ablation exists to make. 10 retries with
# the SDK's capped 8s backoff covers ~60s, i.e. past the reset.
_JUDGE_MAX_RETRIES = int(os.environ.get("GEMINI_JUDGE_MAX_RETRIES", "10"))


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
            max_retries=_JUDGE_MAX_RETRIES,
        ),
        bypass_n=True,
    )
