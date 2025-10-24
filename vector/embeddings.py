# vector/embeddings.py
from sentence_transformers import SentenceTransformer

_model = None

def load_model(model_name="all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def generate_embeddings(text_list, model_name="all-MiniLM-L6-v2"):
    model = load_model(model_name)
    embeddings = model.encode(text_list, convert_to_tensor=False)
    return embeddings.tolist()