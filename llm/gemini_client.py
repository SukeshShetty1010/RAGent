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
from openai import OpenAI

from utils.usage_counter import UsageCounter

logger = logging.getLogger(__name__)

# gemini-2.5-flash returns 404 ("no longer available to new users") for
# accounts created after its cutoff — confirmed live 2026-08-16, both the
# native and OpenAI-compat endpoints. gemini-flash-latest is an alias that
# always resolves to Google's current default flash model, so this stays
# correct across future deprecations without a code change.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_DIM = 768

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
    url = f"{_GEMINI_REST_BASE}/models/{GEMINI_EMBED_MODEL}:batchEmbedContents?key={api_key}"

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
            resp = requests.post(url, json=payload, timeout=30)
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
