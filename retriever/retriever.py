# retriever/retriever.py
from vector.embeddings import generate_embeddings
from vector.weaviate_client import search

def retrieve_similar(query: str, class_name: str = "Games", top_k: int = 5):
    embedding = generate_embeddings([query])[0]
    results = search(embedding, class_name, top_k)
    return results