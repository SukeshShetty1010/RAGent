# ============================================================
# llm/voyage_client.py
# Voyage AI Reranker Client (HTTP)
#
# Replaces the in-process fastembed cross-encoder on the Render
# request path: the ONNX model is CPU-bound on the free tier
# (0.1 vCPU), where a real query's retrieval+rerank stage measured
# ~106-122s live vs ~1-3s locally. This is a plain HTTP call, so no
# local model is loaded and no RAM is spent on it.
#
# Voyage returns NORMALIZED 0..1 relevance scores, not the raw
# ms-marco logits (-7.97..+10.77) that retriever/quality_gate.py's
# floors are calibrated on. The gate's floors are provider-scoped for
# exactly this reason — see quality_gate.py's _FLOORS.
# ============================================================
from __future__ import annotations

import logging
import os
import time
from typing import List

import requests

from utils.usage_counter import UsageCounter

logger = logging.getLogger(__name__)

VOYAGE_RERANK_MODEL = os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")

_VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"

# Explicit connect/read timeouts are load-bearing, not defensive style:
# api/main.py's SSE generator blocks on an unbounded queue.get() and only
# checks request.is_disconnected() *before* each get, so an untimed hang
# in the retrieval thread wedges the stream forever (hit live once, see
# SESSION_NOTES.md 10c).
_TIMEOUT = (5, 10)  # (connect, read) seconds

# One retry, matching llm/gemini_client.py's hand-rolled backoff loop
# (this repo carries no tenacity/backoff dependency). Bounds worst-case
# wall time at roughly 2 * 15s + 1s backoff.
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 1.0

# Voyage's documented per-request document cap.
_MAX_DOCUMENTS = 1000


def _get_voyage_api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    # Same quote-stripping defence as _get_gemini_api_key(): python-dotenv
    # strips quoted values, but `docker run --env-file` and similar raw
    # injection do not, and a literal quote silently 401s every request.
    api_key = api_key.strip().strip('"').strip("'")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    return api_key


def rerank(query: str, documents: List[str]) -> List[float]:
    """Score `documents` against `query` with Voyage's reranker.

    Returns one score per document IN INPUT ORDER. Voyage's response
    `data` is sorted by descending relevance, so scores are re-mapped
    through each entry's `index` field — reading them positionally
    would silently scramble the scores onto the wrong chunks.

    Raises on any failure (bad key, non-200, timeout). Callers own the
    fail-soft behavior, matching how retriever/rag_retriever.py already
    degrades around the local reranker.
    """
    if not documents:
        return []
    if len(documents) > _MAX_DOCUMENTS:
        raise ValueError(
            f"Voyage rerank accepts at most {_MAX_DOCUMENTS} documents, got {len(documents)}"
        )

    api_key = _get_voyage_api_key()
    payload = {
        "query": query,
        "documents": documents,
        "model": VOYAGE_RERANK_MODEL,
        # Voyage truncates over-long documents server-side rather than
        # erroring; ~500-token editorial chunks are well under the limit,
        # but web-search evidence is not length-controlled.
        "truncation": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = None
    for attempt in range(_MAX_RETRIES):
        last_attempt = attempt == _MAX_RETRIES - 1
        try:
            resp = requests.post(
                _VOYAGE_RERANK_URL, json=payload, headers=headers, timeout=_TIMEOUT
            )
        except requests.exceptions.RequestException as exc:
            if last_attempt:
                raise
            logger.warning(
                f"Voyage rerank request failed ({exc}), retrying in "
                f"{_RETRY_BACKOFF_SECONDS}s (attempt {attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(_RETRY_BACKOFF_SECONDS)
            continue

        # Retry only on rate limits and server-side faults — a 400/401 is
        # a config error that a retry cannot fix.
        if (resp.status_code == 429 or resp.status_code >= 500) and not last_attempt:
            logger.warning(
                f"Voyage rerank returned {resp.status_code}, retrying in "
                f"{_RETRY_BACKOFF_SECONDS}s (attempt {attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(_RETRY_BACKOFF_SECONDS)
            continue

        resp.raise_for_status()
        break

    data = resp.json()

    scores: List[float] = [0.0] * len(documents)
    for entry in data["data"]:
        scores[int(entry["index"])] = float(entry["relevance_score"])

    usage = data.get("usage") or {}
    UsageCounter.get().record(
        "voyage",
        "rerank",
        count=1,
        prompt_tokens=int(usage.get("total_tokens", 0)),
    )

    return scores
