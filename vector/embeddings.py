# vector/embeddings.py
from sentence_transformers import SentenceTransformer
from utils.gpu_utils import get_device

_model = None

def load_model(model_name="all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        device = get_device()
        print(f"[embeddings.py] Loading SentenceTransformer '{model_name}' on {device.upper()}")
        _model = SentenceTransformer(model_name, device=device)
    return _model

def generate_embeddings(
    text_list,
    model_name="all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress_bar: bool = False
):
    """
    Generate embeddings in batch on GPU.
    """
    if not text_list:
        return []
    
    model = load_model(model_name)
    embeddings = model.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_tensor=False,
        normalize_embeddings=True  # Critical for cosine similarity
    )
    return embeddings.tolist()