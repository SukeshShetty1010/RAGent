# llm/modal_embed.py

import modal
from typing import List

# ------------------------------------------------------------------------------
# Modal App
# ------------------------------------------------------------------------------
app = modal.App("e5-base-v2-embeddings")

# ------------------------------------------------------------------------------
# Model constants
# ------------------------------------------------------------------------------
MODEL_NAME = "intfloat/e5-base-v2"
MODEL_DIR = "/models/e5-base-v2"

# ------------------------------------------------------------------------------
# Image definition
# ------------------------------------------------------------------------------
def download_model():
    """
    Download model weights at IMAGE BUILD time so they are baked
    into the container and not re-downloaded on cold start.
    """
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_NAME, cache_folder=MODEL_DIR)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "sentence-transformers>=2.6.0",
        "transformers>=4.45.0",
        "huggingface-hub>=0.25.0",
    )
    .run_function(download_model)
)

# ------------------------------------------------------------------------------
# Embedding Service (Class-based, warm containers)
# ------------------------------------------------------------------------------
@app.cls(
    gpu="T4",
    image=image,
    timeout=600,
    container_idle_timeout=300,
    max_containers=5,
)
class E5Embedder:
    """
    Persistent embedding service using intfloat/e5-base-v2.
    """

    def __enter__(self):
        """
        Runs once per container when it starts.
        Loads the model into memory.
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=MODEL_DIR,
            device="cuda",
        )

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def embed_text(
        self,
        texts: List[str],
        *,
        mode: str = "passage",
    ) -> List[List[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of input strings
            mode: "passage" (default) or "query"

        Returns:
            List of normalized embedding vectors (float lists)
        """

        if mode not in {"passage", "query"}:
            raise ValueError("mode must be either 'passage' or 'query'")

        prefixed = [f"{mode}: {t}" for t in texts]

        embeddings = self.model.encode(
            prefixed,
            batch_size=16,
            normalize_embeddings=True,  # REQUIRED for cosine similarity
            show_progress_bar=False,
        )

        return embeddings.tolist()


# ------------------------------------------------------------------------------
# Optional local test (Modal run)
# ------------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    embedder = E5Embedder()

    vectors = embedder.embed_text.remote(
        [
            "Far Cry 5 is an open-world first-person shooter set in Montana.",
            "The game emphasizes exploration and player choice.",
        ],
        mode="passage",
    )

    print(f"Returned {len(vectors)} embeddings")
    print(f"Vector dimension: {len(vectors[0])}")
