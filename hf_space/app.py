"""
hf_space/app.py

Cross-encoder reranking service, deployed as a Hugging Face Docker Space
(free CPU Basic: 2 vCPU / 16GB).

Why this exists: the same reranker running in-process on Render's free
tier (0.1 vCPU / 512MB) measured ~106-122s per query and needed
batch_size=1 to avoid OOM. Moving it here trades that for an HTTP call
against 20x the CPU and 32x the RAM, at no cost and with no rate limits.

Deliberately runs the IDENTICAL model as the in-process path
(Xenova/ms-marco-MiniLM-L-6-v2). That is the whole point: scores stay on
the same raw-logit scale, so RAGent's quality_gate floors -- calibrated
in evaluation/results/relevance_calibration_2026-08-12.json -- remain
valid with no re-calibration.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rerank_space")

MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

# Optional shared secret. A private Space is already gated by the HF
# bearer token, so this is a second, independent lock — set RERANK_SECRET
# in the Space's Settings > Secrets and RAGent sends it as X-Rerank-Key.
RERANK_SECRET = os.environ.get("RERANK_SECRET", "").strip()

# 16GB of RAM here, versus the 512MB that forced batch_size=1 on Render.
# The fastembed default (64) is fine; kept explicit so the contrast with
# retriever/rag_retriever.py's _RERANK_BATCH_SIZE is legible.
BATCH_SIZE = 64

# Guardrail against an accidental unbounded request. RAGent's largest
# real shape is 40 candidates (LISTICLE: limit=10 -> fetch_limit=40).
MAX_DOCUMENTS = 200

# Eager load at import, same rule as retriever/rag_retriever.py: a model
# failure must be a visible boot error, not a hang on the first request.
logger.info(f"Loading cross-encoder {MODEL_NAME} ...")
_t0 = time.time()
reranker = TextCrossEncoder(model_name=MODEL_NAME)
logger.info(f"Cross-encoder ready in {time.time() - _t0:.2f}s")

app = FastAPI(title="RAGent Reranker", version="1.0.0")


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: List[str]


class RerankResponse(BaseModel):
    scores: List[float]
    model: str
    elapsed_ms: int


def _check_auth(provided: Optional[str]) -> None:
    if not RERANK_SECRET:
        return
    if provided != RERANK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Rerank-Key")


@app.get("/health")
def health() -> dict:
    """Unauthenticated, and cheap enough to be the keepalive target.

    Reports the model as a liveness signal rather than just returning
    200 — a container that booted but failed to load the model would
    otherwise look healthy to the cron.
    """
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest, x_rerank_key: Optional[str] = Header(default=None)) -> RerankResponse:
    """Score every document against the query.

    Returns scores IN INPUT ORDER — the caller zips them straight onto
    its candidate list, so any reordering here would silently attach
    scores to the wrong chunks.

    Scores are raw ms-marco logits (roughly -8..+11), NOT normalized.
    Do not "helpfully" sigmoid them: RAGent's refusal floors are
    calibrated on the raw scale.
    """
    _check_auth(x_rerank_key)

    if not req.documents:
        return RerankResponse(scores=[], model=MODEL_NAME, elapsed_ms=0)
    if len(req.documents) > MAX_DOCUMENTS:
        raise HTTPException(
            status_code=413,
            detail=f"Too many documents: {len(req.documents)} > {MAX_DOCUMENTS}",
        )

    t0 = time.time()
    scores = [float(s) for s in reranker.rerank(req.query, req.documents, batch_size=BATCH_SIZE)]
    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info(f"reranked {len(req.documents)} docs in {elapsed_ms}ms")

    return RerankResponse(scores=scores, model=MODEL_NAME, elapsed_ms=elapsed_ms)
