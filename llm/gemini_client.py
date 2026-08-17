# ============================================================
# llm/gemini_client.py
# Gemini Client Wrapper (chat via OpenAI-compat endpoint,
# embeddings via native REST — embeddings are not exposed on
# Gemini's OpenAI-compat surface, so they need a separate call).
# ============================================================
from __future__ import annotations
import logging
import math
import os
import time
from typing import Any, List

import requests
from openai import BadRequestError, OpenAI

from utils.usage_counter import UsageCounter

logger = logging.getLogger(__name__)

# gemini-2.5-flash returns 404 ("no longer available to new users") for
# accounts created after its cutoff — confirmed live 2026-08-16, both the
# native and OpenAI-compat endpoints. gemini-flash-latest is an alias that
# always resolves to Google's current default flash model, so this stays
# correct across future deprecations without a code change.
# gemini-flash-latest is deliberately NOT the default. That alias
# currently resolves to gemini-3.7-flash, which allows 20 requests/day on
# the free tier and was observed closing long response streams
# mid-answer, with no finish_reason, leaving a severed sentence on screen
# (2026-08-18). gemini-flash-lite-latest streamed to completion in every
# measured run (7/7, finish_reason="stop") and has a far larger free
# allowance. A smaller model that finishes its answer beats a stronger
# one that gets cut off. Override with GEMINI_MODEL if a paid key or a
# different tradeoff applies.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_DIM = 768

# Suppress the thinking budget on models that support it. Set to "" to
# stop sending the parameter entirely.
#
# This is NOT uniformly supported, which quietly broke the promise that
# GEMINI_MODEL is swappable without a code change: gemini-flash-latest
# (currently gemini-3.7-flash) accepts "none", while every lite model
# rejects it with a bare 400 "Request contains an invalid argument"
# (confirmed live 2026-08-18 for gemini-flash-lite-latest and
# gemini-3.5-flash-lite; both accept "low"). Because the clients treat
# any Gemini exception as a reason to fall back, pointing GEMINI_MODEL at
# a lite model made *every* request fail over to Groq while looking like
# a healthy Gemini-primary deploy. create_completion() below degrades
# instead.
GEMINI_REASONING_EFFORT = os.environ.get("GEMINI_REASONING_EFFORT", "none").strip()

# model name -> whether it accepts reasoning_effort. Populated on first
# rejection so the retry is paid once per process, not once per request.
_reasoning_effort_supported: dict[str, bool] = {}

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"

# batchEmbedContents caps at 100 requests per call.
_EMBED_BATCH_SIZE = 100
_EMBED_MAX_RETRIES = 4

_gemini_client: OpenAI | None = None


def _get_gemini_client() -> OpenAI:
    """Lazy OpenAI-compat client for Gemini chat completions.

    Mirrors _get_groq_client()'s shape exactly: raises ValueError if the
    key is unset, so callers' existing except-and-fallback path handles
    it without any new error-handling branch.
    """
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = OpenAI(base_url=_GEMINI_BASE_URL, api_key=_get_gemini_api_key())
    return _gemini_client


def create_completion(client: OpenAI, *, model: str, **kwargs: Any) -> Any:
    """chat.completions.create with reasoning_effort applied only where
    the model accepts it.

    Sends reasoning_effort on the first call for a given model; if that
    call is rejected as a bad request, retries once without it and
    remembers the answer. The flag is only marked unsupported when the
    retry actually succeeds, so a 400 caused by something else (an
    oversized prompt, say) still surfaces as itself rather than being
    permanently misattributed to reasoning_effort.

    Used for streaming and non-streaming alike. With stream=True the
    request is issued eagerly and a rejection surfaces here, before any
    token has been yielded, so the retry cannot duplicate output.
    """
    effort = GEMINI_REASONING_EFFORT
    if effort and _reasoning_effort_supported.get(model, True):
        try:
            return client.chat.completions.create(
                model=model, reasoning_effort=effort, **kwargs
            )
        except BadRequestError:
            result = client.chat.completions.create(model=model, **kwargs)
            _reasoning_effort_supported[model] = False
            logger.warning(
                f"{model} rejected reasoning_effort={effort!r}; "
                f"continuing without it for the rest of this process"
            )
            return result
    return client.chat.completions.create(model=model, **kwargs)


def _get_gemini_api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    # Strip surrounding whitespace/quotes -- python-dotenv strips quoted
    # values automatically, but env vars supplied via `docker run
    # --env-file` or similar raw injection do not, and a literal quote
    # in the key silently turns every request into a 400/401.
    api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return api_key


def _l2_normalize(vector: List[float]) -> List[float]:
    """Required by Google for any embedContent outputDimensionality < 3072 —
    only the full 3072-dim output is pre-normalized server-side."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def embed_texts(
    texts: List[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> List[List[float]]:
    """Embed a batch of texts with Gemini, 768-dim, L2-normalized.

    task_type: "RETRIEVAL_DOCUMENT" for ingest-time chunk embeddings,
    "RETRIEVAL_QUERY" for query-time embeddings. Batches internally at
    the API's 100-request batchEmbedContents cap and retries with
    exponential backoff on 429s.
    """
    if not texts:
        return []
    api_key = _get_gemini_api_key()
    url = f"{_GEMINI_REST_BASE}/models/{GEMINI_EMBED_MODEL}:batchEmbedContents"
    # Key goes in a header, NOT ?key=... — requests' HTTPError message
    # embeds the full URL, so a query-param key ends up verbatim in every
    # 429/500 traceback, in Render's logs and in local run logs. Hit for
    # real during the embedding migration's 429 wall.
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    results: List[List[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        payload = {
            "requests": [
                {
                    "model": f"models/{GEMINI_EMBED_MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": GEMINI_EMBED_DIM,
                }
                for text in batch
            ]
        }

        delay = 1.0
        for attempt in range(_EMBED_MAX_RETRIES):
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 429 and attempt < _EMBED_MAX_RETRIES - 1:
                logger.warning(
                    f"Gemini embedContent rate-limited, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{_EMBED_MAX_RETRIES})"
                )
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            break

        data = resp.json()
        for embedding in data["embeddings"]:
            results.append(_l2_normalize(embedding["values"]))

        UsageCounter.get().record("gemini", "embedding", count=len(batch))

    return results


def embed_text(text: str, *, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    return embed_texts([text], task_type=task_type)[0]
