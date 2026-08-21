"""
retriever/reranker_provider.py

Single source of truth for which reranker backend is active.

Deliberately dependency-free: retriever/quality_gate.py needs the
provider name to pick its relevance floors, but importing
retriever/rag_retriever.py to get it would drag the fastembed ONNX
models into every hermetic unit test that touches the quality gate.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

RerankerProvider = Literal["local", "hfspace", "cloudflare", "voyage"]

VALID_PROVIDERS = ("local", "hfspace", "cloudflare", "voyage")

# "local"      = in-process fastembed cross-encoder (Xenova/ms-marco-MiniLM-L-6-v2)
# "hfspace"    = the SAME model over HTTP, hosted on an HF Docker Space
#                (hf_space/) — identical raw-logit scale, so it shares
#                "local"'s calibrated floors. NOT currently deployed:
#                HF moved Docker Spaces behind PRO in July 2026.
# "cloudflare" = @cf/baai/bge-reranker-base on Workers AI — free tier,
#                always warm, but a DIFFERENT model and therefore a
#                different scale
# "voyage"     = Voyage's rerank HTTP API — also a different model,
#                normalized 0..1
# Only same-model providers may share floors. See quality_gate.py's
# _FLOORS.
DEFAULT_PROVIDER: RerankerProvider = "local"


def resolve_reranker_provider() -> str:
    """Read RERANKER_PROVIDER from the environment, normalized.

    Read on every call rather than cached at import so tests can flip it
    with monkeypatch.setenv without reloading modules. An unrecognized
    value falls back to the default rather than raising — a typo'd env
    var must not take the whole service down.
    """
    raw = (os.environ.get("RERANKER_PROVIDER") or "").strip().strip('"').strip("'").lower()
    if raw not in VALID_PROVIDERS:
        return DEFAULT_PROVIDER
    return raw


# Human-readable rerank_score scale, per provider, for quoting inside LLM
# prompts (see agent/decisions/web_search_decision.py). Providers sharing
# a model share a description, same as they share quality_gate.py floors.
_SCALE_DESCRIPTIONS: dict[str, str] = {
    "local": "cross-encoder logit, unbounded — higher is better, roughly: <0 weak, >3 strong",
    "hfspace": "cross-encoder logit, unbounded — higher is better, roughly: <0 weak, >3 strong",
    "cloudflare": "normalized 0.0-1.0 — higher is better",
    "voyage": "normalized 0.0-1.0 — higher is better",
}


def describe_score_scale(provider: Optional[str] = None) -> str:
    """Short description of the active (or given) provider's rerank_score
    scale, for embedding in an LLM prompt. A prompt that hardcodes one
    provider's scale silently misdescribes another once the active
    provider changes — this keeps the description provider-aware instead.
    """
    p = provider if provider in VALID_PROVIDERS else resolve_reranker_provider()
    return _SCALE_DESCRIPTIONS.get(p, "provider-specific scale — see retriever/quality_gate.py")
