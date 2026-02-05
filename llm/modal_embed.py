"""
Modal-hosted embedding service for Stage 5: Editorial Chunk Embeddings

Model:
- intfloat/e5-base-v2

GPU:
- NVIDIA T4

Design guarantees:
- Runs on Modal
- Uses env-based Modal auth (implicit)
- Model loaded once per container
- Modal 1.0 compliant (no deprecated APIs)
"""

from __future__ import annotations
from typing import List

import modal

# ------------------------------------------------------------------------------
# Modal App
# ------------------------------------------------------------------------------

app = modal.App("editorial-embedding-service")

# ------------------------------------------------------------------------------
# Image
# ------------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sentencepiece",
        "numpy",
    )
)

# ------------------------------------------------------------------------------
# HF Cache Volume
# ------------------------------------------------------------------------------

model_volume = modal.Volume.from_name(
    "hf-e5-embeddings",
    create_if_missing=True,
)

# ------------------------------------------------------------------------------
# Stateful Embedder (Modal 1.0 compliant)
# ------------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="T4",
    timeout=60 * 5,
    volumes={"/models": model_volume},
    scaledown_window=300,  # Modal 1.0 replacement for container_idle_timeout
)
@modal.concurrent(max_inputs=8)  # Modal 1.0 replacement for allow_concurrent_inputs
class E5Embedder:
    @modal.enter()
    def load_model(self):
        """
        Runs ONCE per container lifecycle.
        Guaranteed by Modal.
        """
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.device = "cuda"
        model_id = "intfloat/e5-base-v2"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir="/models",
        )

        self.model = AutoModel.from_pretrained(
            model_id,
            cache_dir="/models",
        ).to(self.device)

        self.model.eval()

    @modal.method()
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate normalized E5 embeddings.
        """
        import torch

        if not texts:
            return []

        prefixed = [f"passage: {t}" for t in texts]

        with torch.no_grad():
            encoded = self.tokenizer(
                prefixed,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            output = self.model(**encoded)

            embeddings = self._mean_pool(
                output.last_hidden_state,
                encoded["attention_mask"],
            )

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1,
            )

        return embeddings.cpu().tolist()

    @staticmethod
    def _mean_pool(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


# ------------------------------------------------------------------------------
# Entrypoint (smoke test only)
# ------------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("🚀 E5 embedding service ready (Modal 1.0).")
