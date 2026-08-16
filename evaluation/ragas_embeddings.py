"""
evaluation/ragas_embeddings.py

Thin RAGAS embeddings adapter over llm/gemini_client.py's Gemini
embedding API, which retriever/rag_retriever.py also uses for dense
retrieval. RAGAS's `answer_relevancy` metric needs an embeddings
backend; this reuses the existing Gemini embedding call instead of
loading sentence-transformers/torch locally (forbidden on the Render
path, and this module is also imported from evaluation scripts that
should stay lightweight).
"""

from __future__ import annotations

import asyncio
from typing import List

from ragas.embeddings import BaseRagasEmbeddings

from llm.gemini_client import embed_text, embed_texts


class GeminiEmbeddings(BaseRagasEmbeddings):
    """RAGAS embeddings backend backed by Gemini's embedContent API."""

    def embed_query(self, text: str) -> List[float]:
        return embed_text(text, task_type="RETRIEVAL_QUERY")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    async def aembed_query(self, text: str) -> List[float]:
        # RAGAS runs under asyncio; the Gemini client is a synchronous
        # requests call, so offload it to a thread to avoid blocking
        # the event loop.
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self.embed_documents, texts)
