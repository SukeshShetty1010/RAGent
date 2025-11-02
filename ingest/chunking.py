# ingest/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Chunks documents using character-based splitter.
    GPU not involved — this is pure text processing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata = chunk.metadata.copy()  # Avoid mutation
            chunk.metadata['chunk_id'] = i
            all_chunks.append(chunk)
    return all_chunks