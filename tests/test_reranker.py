"""
tests/test_reranker.py

Sanity checks for the in-process fastembed cross-encoder reranker
(retriever/rag_retriever.py's `reranker`), which replaced the
Modal-hosted CrossEncoderReranker. Full calibration parity against
evaluation/results/relevance_calibration_2026-08-12.json requires a
live Qdrant corpus and is a Phase 7 manual verification step, not a
unit test -- this only checks the model identity and basic ordering
sanity, both fast and offline once the ONNX model is cached.
"""

import pytest

pytestmark = pytest.mark.unit


def test_reranker_model_identity():
    from retriever.rag_retriever import reranker
    assert reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_reranker_orders_relevant_above_irrelevant():
    from retriever.rag_retriever import reranker

    query = "When was The Legend of Zelda: Breath of the Wild released?"
    relevant = "The Legend of Zelda: Breath of the Wild was released on March 3, 2017 for the Nintendo Switch."
    irrelevant = "The recipe calls for two cups of flour and a teaspoon of baking soda."

    scores = list(reranker.rerank(query, [irrelevant, relevant]))
    assert scores[1] > scores[0]
