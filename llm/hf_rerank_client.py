# ============================================================
# llm/hf_rerank_client.py
# Hugging Face Space Reranker Client (HTTP)
#
# Talks to the Docker Space in hf_space/, which runs the SAME model as
# the in-process path (Xenova/ms-marco-MiniLM-L-6-v2) on free CPU Basic
# hardware (2 vCPU / 16GB) instead of Render's throttled 0.1 vCPU.
#
# Because the model is identical, scores stay on the raw-logit scale
# that retriever/quality_gate.py's floors are calibrated on — this
# provider needs no re-calibration, unlike a swap to a different
# reranker (see _FLOORS there).
# ============================================================
from __future__ import annotations

import logging
import os
import time
from typing import List

import requests

from utils.usage_counter import UsageCounter

logger = logging.getLogger(__name__)

# Base URL of the Space, e.g. https://<user>-ragent-reranker.hf.space
HF_RERANK_URL = os.environ.get("HF_RERANK_URL", "").strip().strip('"').strip("'")

# Read timeout is generous relative to the ~1-3s expected: a free Space
# that has been restarted by HF must reload its model on the first
# request. Still explicitly bounded — api/main.py's SSE generator blocks
# on an unbounded queue.get(), so an untimed call wedges the stream
# forever (hit live once, see SESSION_NOTES.md 10c).
_TIMEOUT = (5, 45)  # (connect, read) seconds

# One retry, matching llm/gemini_client.py's hand-rolled backoff loop
# (no tenacity/backoff dependency in this repo).
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 1.0


def _get_base_url() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    url = os.environ.get("HF_RERANK_URL", "")
    # Same quote-stripping defence as the Gemini/Voyage key getters:
    # `docker run --env-file` passes literal quotes through, which here
    # would produce an unresolvable hostname on every request.
    url = url.strip().strip('"').strip("'").rstrip("/")
    if not url:
        raise ValueError("HF_RERANK_URL environment variable not set")
    return url


def _headers() -> dict:
    from dotenv import load_dotenv
    load_dotenv()
    headers = {"Content-Type": "application/json"}

    # Shared secret enforced by the Space itself (hf_space/app.py).
    secret = (os.environ.get("HF_RERANK_SECRET") or "").strip().strip('"').strip("'")
    if secret:
        headers["X-Rerank-Key"] = secret

    # Required only when the Space is private.
    token = (os.environ.get("HF_TOKEN") or "").strip().strip('"').strip("'")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def health() -> dict:
    """Liveness probe against the Space. Used by verification scripts and
    the keepalive cron's local equivalent, not by the request path."""
    resp = requests.get(f"{_get_base_url()}/health", headers=_headers(), timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def rerank(query: str, documents: List[str]) -> List[float]:
    """Score `documents` against `query`, returned in input order.

    The Space returns raw ms-marco logits (roughly -8..+11) — the same
    scale as the in-process reranker, deliberately.

    Raises on any failure (unreachable Space, non-200, timeout).
    Callers own the fail-soft behavior, matching how
    retriever/rag_retriever.py already degrades around the local path.
    """
    if not documents:
        return []

    url = f"{_get_base_url()}/rerank"
    headers = _headers()
    payload = {"query": query, "documents": documents}

    resp = None
    for attempt in range(_MAX_RETRIES):
        last_attempt = attempt == _MAX_RETRIES - 1
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            if last_attempt:
                raise
            logger.warning(
                f"HF rerank request failed ({exc}), retrying in "
                f"{_RETRY_BACKOFF_SECONDS}s (attempt {attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(_RETRY_BACKOFF_SECONDS)
            continue

        # A 503 is the normal signal that a slept Space is waking up, so
        # it is worth exactly one retry. A 401/413 is a config error that
        # retrying cannot fix.
        if resp.status_code >= 500 and not last_attempt:
            logger.warning(
                f"HF rerank returned {resp.status_code} (Space may be waking), retrying in "
                f"{_RETRY_BACKOFF_SECONDS}s (attempt {attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(_RETRY_BACKOFF_SECONDS)
            continue

        resp.raise_for_status()
        break

    data = resp.json()
    scores = [float(s) for s in data["scores"]]

    if len(scores) != len(documents):
        raise ValueError(
            f"HF rerank returned {len(scores)} scores for {len(documents)} documents"
        )

    # No token accounting here — the Space is fixed-cost free hardware,
    # so only the request count is meaningful.
    UsageCounter.get().record("hfspace", "rerank", count=1)

    return scores
