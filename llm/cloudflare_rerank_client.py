# ============================================================
# llm/cloudflare_rerank_client.py
# Cloudflare Workers AI Reranker Client (HTTP)
#
# Runs @cf/baai/bge-reranker-base on Cloudflare's edge instead of the
# in-process ONNX cross-encoder, which is CPU-bound on Render's free
# tier (~106-122s per query at 0.1 vCPU, measured live).
#
# Free allocation is 10,000 Neurons/day with no payment method on file,
# and the endpoint is always warm — no cold start, no keepalive.
#
# IMPORTANT: this is a DIFFERENT model from the in-process reranker, so
# its scores are NOT on the scale retriever/quality_gate.py's local
# floors are calibrated against. Cloudflare's docs say the score "can be
# mapped to [0,1] by sigmoid function", implying raw logits, but it
# applies the sigmoid server-side — confirmed live 2026-08-17, a 4-doc
# probe returned 0.9998979 for the match and 3.74e-05 for the misses.
# So the scale is 0..1 like Voyage's, and heavily saturated at both
# ends, which will matter when floors are picked: the usable signal
# sits in the tails, not around 0.5. _FLOORS["cloudflare"] stays None
# until evaluation/calibrate_relevance.py has been re-run.
#
# Measured cost at RAGent's real shapes (10,000 Neurons/day free):
#   20 candidates ~10,240 tokens = 2.89 neurons -> ~3,455 queries/day
#   40 candidates ~20,480 tokens = 5.79 neurons -> ~1,727 queries/day
# ============================================================
from __future__ import annotations

import logging
import os
import time
from typing import List

import requests

from utils.usage_counter import UsageCounter

logger = logging.getLogger(__name__)

CLOUDFLARE_RERANK_MODEL = os.environ.get(
    "CLOUDFLARE_RERANK_MODEL", "@cf/baai/bge-reranker-base"
)

_TIMEOUT = (5, 20)  # (connect, read) seconds — bounded; see api/main.py's SSE queue
_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 1.0


def _env(name: str) -> str:
    from dotenv import load_dotenv
    load_dotenv()
    value = os.environ.get(name, "")
    # Same quote-stripping defence as the Gemini/Voyage key getters:
    # `docker run --env-file` passes literal quotes straight through.
    value = value.strip().strip('"').strip("'")
    if not value:
        raise ValueError(f"{name} environment variable not set")
    return value


def _endpoint() -> str:
    account_id = _env("CLOUDFLARE_ACCOUNT_ID")
    return (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/{CLOUDFLARE_RERANK_MODEL}"
    )


def _parse_scores(data: dict, expected: int) -> List[float]:
    """Map Cloudflare's response back onto input order.

    Confirmed live: the response IS sorted by score descending, and each
    entry carries an `id` naming its position in the request. Reading it
    positionally would therefore attach every score to the wrong chunk
    while still producing plausible-looking numbers, so the id/index
    field is authoritative and the positional fallback only applies if
    the shape ever loses that field entirely.
    """
    result = data.get("result") or {}
    entries = result.get("response")
    if entries is None:
        raise ValueError(f"Unexpected Cloudflare rerank response: {str(data)[:300]}")

    scores = [0.0] * expected
    seen = 0
    for position, entry in enumerate(entries):
        if isinstance(entry, dict):
            index = entry.get("id", entry.get("index"))
            score = entry.get("score")
        else:  # a bare number, if the shape ever changes
            index, score = None, entry

        if index is None:
            index = position
        index = int(index)
        if not 0 <= index < expected:
            raise ValueError(f"Cloudflare rerank returned out-of-range index {index}")

        scores[index] = float(score)
        seen += 1

    if seen != expected:
        raise ValueError(
            f"Cloudflare rerank scored {seen} of {expected} documents "
            "(top_k must not be set for this use)"
        )
    return scores


def rerank(query: str, documents: List[str]) -> List[float]:
    """Score `documents` against `query`, returned in input order.

    Raises on any failure. Callers own the fail-soft behavior, matching
    how retriever/rag_retriever.py already degrades around the local path.
    """
    if not documents:
        return []

    url = _endpoint()
    headers = {
        "Authorization": f"Bearer {_env('CLOUDFLARE_API_TOKEN')}",
        "Content-Type": "application/json",
    }
    # top_k is deliberately omitted: every candidate needs a score, since
    # the caller zips them onto its full candidate list.
    payload = {"query": query, "contexts": [{"text": d} for d in documents]}

    resp = None
    for attempt in range(_MAX_RETRIES):
        last_attempt = attempt == _MAX_RETRIES - 1
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            if last_attempt:
                raise
            logger.warning(
                f"Cloudflare rerank request failed ({exc}), retrying in "
                f"{_RETRY_BACKOFF_SECONDS}s (attempt {attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(_RETRY_BACKOFF_SECONDS)
            continue

        # Retry rate limits and server faults only — a 400/403 is a config
        # error, and a daily-allocation 429 will not clear within a retry
        # either, but one cheap attempt costs less than a failed query.
        if (resp.status_code == 429 or resp.status_code >= 500) and not last_attempt:
            logger.warning(
                f"Cloudflare rerank returned {resp.status_code}, retrying in "
                f"{_RETRY_BACKOFF_SECONDS}s (attempt {attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(_RETRY_BACKOFF_SECONDS)
            continue

        resp.raise_for_status()
        break

    data = resp.json()
    if not data.get("success", True):
        raise ValueError(f"Cloudflare rerank reported failure: {str(data.get('errors'))[:300]}")

    scores = _parse_scores(data, len(documents))

    # The response reports both tokens and the Neurons they cost (the
    # actually-budgeted resource, 10k/day free). Tokens go into the
    # existing counter schema; the neuron figure is logged rather than
    # given a new stored field, since it is derivable from tokens at a
    # measured ~0.000283 neurons/token.
    usage = (data.get("result") or {}).get("usage") or {}
    neurons = usage.get("neurons")
    if neurons is not None:
        logger.info(
            f"Cloudflare rerank: {len(documents)} docs, "
            f"{usage.get('total_tokens', 0)} tokens, {float(neurons):.3f} neurons"
        )
    UsageCounter.get().record(
        "cloudflare",
        "rerank",
        count=1,
        prompt_tokens=int(usage.get("total_tokens", 0)),
    )

    return scores
