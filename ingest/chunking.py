# ingest/chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Chunks the documents using the specified parameters.
    
    Args:
        docs (List[Document]): List of documents to chunk.
    
    Returns:
        List[Document]: List of chunked documents with updated metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len  # Using character length as approximation for tokens
    )
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            all_chunks.append(chunk)
    return all_chunks