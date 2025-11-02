# vector/embed.py
from langchain_huggingface import HuggingFaceEmbeddings
from utils.gpu_utils import get_device

def get_embedding_model():
    """
    Returns GPU-accelerated HuggingFace embedding model.
    """
    device = get_device()
    print(f"[embed.py] Loading HuggingFaceEmbeddings on {device.upper()}")
    
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )