# vector/embed.py
"""
Unified embedding module used by ALL ingestion components.

This wraps HuggingFaceEmbeddings (LangChain) AND exposes embed_texts(texts)
in a stable way so ingest/upsert.py can always call:
    
    from vector.embed import embed_texts

Features:
 - GPU-aware (uses utils/gpu_utils.get_device)
 - Normalizes vectors (important for Weaviate + MiniLM)
 - Automatic batching
 - Supports embed_documents() or embed_query(), depending on wrapper availability
"""

from langchain_huggingface import HuggingFaceEmbeddings
from utils.gpu_utils import get_device


# -----------------------------
# Model Loader
# -----------------------------
_EMBED_MODEL = None


def get_embedding_model():
    """
    Lazily loads the HuggingFace embedding model (GPU or CPU).

    Uses: sentence-transformers/all-MiniLM-L6-v2
    """
    global _EMBED_MODEL

    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    device = get_device()
    print(f"[embed.py] Loading HuggingFaceEmbeddings on {device.upper()}")

    _EMBED_MODEL = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    return _EMBED_MODEL


# -----------------------------
# Main embedding function (Used by upsert.py)
# -----------------------------
def embed_texts(texts, batch_size: int = 64):
    """
    Convert list[str] → list[list[float]] using the selected model.

    This API is what ingest/upsert.py calls.

    It supports:
      - model.embed_documents()  (main path)
      - model.embed_query()      (fallback)
      - model.encode()           (rare fallback)
    """
    if not texts:
        return []

    model = get_embedding_model()

    # --- Fast path: LangChain HF wrapper ---
    if hasattr(model, "embed_documents"):
        # This already handles batching internally.
        return model.embed_documents(texts)

    # --- Fallback: per-item query function (slower) ---
    if hasattr(model, "embed_query"):
        return [model.embed_query(t) for t in texts]

    # --- Rare fallback: SentenceTransformer-like API ---
    if hasattr(model, "encode"):
        arr = model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in arr]

    # --- REALLY rare fallback: HF wrapper internal encoder ---
    if hasattr(model, "client") and hasattr(model.client, "model"):
        encoder = model.client.model
        try:
            if hasattr(encoder, "encode"):
                arr = encoder.encode(texts, convert_to_numpy=True)
                return [v.tolist() for v in arr]
        except Exception:
            pass

    raise RuntimeError(
        "embed_texts: could not find a usable embedding API on HuggingFaceEmbeddings wrapper."
    )
