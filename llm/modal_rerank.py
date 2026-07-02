"""
Modal-hosted cross-encoder reranking service for RAGent

Model:
- cross-encoder/ms-marco-MiniLM-L-6-v2

Compute:
- CPU only (model is ~80MB, no GPU needed)

Design guarantees:
- Runs on Modal, keeping torch/sentence-transformers off the Render
  backend process — loading them there (alongside fastembed's BM25
  encoder) exceeded Render's 512MB free-tier RAM limit.
- Model loaded once per container via @modal.enter.
- Model weights persisted on a Modal Volume to avoid repeated HF
  downloads on cold starts.
"""

from __future__ import annotations
from typing import List

import modal

# ------------------------------------------------------------------------------
# Modal App
# ------------------------------------------------------------------------------

app = modal.App("cross-encoder-rerank-service")

# ------------------------------------------------------------------------------
# Image
# ------------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "sentence-transformers",
    )
)

# ------------------------------------------------------------------------------
# HF Cache Volume
# ------------------------------------------------------------------------------

model_volume = modal.Volume.from_name(
    "hf-cross-encoder-rerank",
    create_if_missing=True,
)

# ------------------------------------------------------------------------------
# Stateful Reranker
# ------------------------------------------------------------------------------

@app.cls(
    image=image,
    timeout=60 * 2,
    volumes={"/models": model_volume},
    scaledown_window=300,
)
@modal.concurrent(max_inputs=8)
class CrossEncoderReranker:
    @modal.enter()
    def load_model(self):
        """
        Runs ONCE per container lifecycle. Guaranteed by Modal.
        """
        from sentence_transformers import CrossEncoder

        self.encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            cache_folder="/models",
        )

    @modal.method()
    def rerank(self, query: str, contents: List[str]) -> List[float]:
        """
        Score each (query, content) pair. Returns one relevance score
        per content, in the same order as the input list.
        """
        if not contents:
            return []

        pairs = [(query, c) for c in contents]
        scores = self.encoder.predict(pairs)
        return [float(s) for s in scores]


# ------------------------------------------------------------------------------
# Entrypoint (smoke test only)
# ------------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    reranker = CrossEncoderReranker()
    scores = reranker.rerank.remote(
        "best open world RPG",
        [
            "Elden Ring is an open world action RPG.",
            "Tetris is a falling-block puzzle game.",
        ],
    )
    print(f"Scores: {scores}")
