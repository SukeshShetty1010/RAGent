"""
evaluation/ragas_embeddings.py

Thin RAGAS embeddings adapter over the Modal-hosted E5Embedder that
retriever/rag_retriever.py already uses for dense retrieval
(`_get_dense_encoder()`). RAGAS's `answer_relevancy` metric needs an
embeddings backend; this reuses the existing lazy Modal binding
instead of loading sentence-transformers/torch locally (forbidden on
the Render path, and this module is also imported from evaluation
scripts that should stay lightweight).
"""

from __future__ import annotations

from typing import List

from ragas.embeddings import BaseRagasEmbeddings

from retriever.rag_retriever import _get_dense_encoder


class ModalE5Embeddings(BaseRagasEmbeddings):
    """RAGAS embeddings backend backed by the Modal E5Embedder service."""

    def embed_query(self, text: str) -> List[float]:
        return _get_dense_encoder().embed_texts.remote([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return _get_dense_encoder().embed_texts.remote(texts)

    async def aembed_query(self, text: str) -> List[float]:
        # RAGAS runs under asyncio; use Modal's native async entrypoint
        # instead of blocking .remote() to avoid Modal's AsyncUsageWarning.
        result = await _get_dense_encoder().embed_texts.remote.aio([text])
        return result[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await _get_dense_encoder().embed_texts.remote.aio(texts)
