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
from typing import Literal

RerankerProvider = Literal["local", "hfspace", "voyage"]

VALID_PROVIDERS = ("local", "hfspace", "voyage")

# "local"   = in-process fastembed cross-encoder (Xenova/ms-marco-MiniLM-L-6-v2)
# "hfspace" = the SAME model over HTTP, hosted on a free HF Docker Space
#             (hf_space/) — identical raw-logit scale, so it shares
#             "local"'s calibrated floors
# "voyage"  = Voyage's rerank HTTP API — normalized 0..1, a different
#             scale that needs its own calibration
# See quality_gate.py's _FLOORS.
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
