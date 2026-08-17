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

from retriever.reranker_provider import resolve_reranker_provider

pytestmark = pytest.mark.unit

# The local cross-encoder only exists when RERANKER_PROVIDER is "local"
# (see retriever/rag_retriever.py) — under "voyage" it is deliberately
# never constructed, which is what keeps the ~300MB ONNX model out of
# the Render process. These tests are about that model specifically, so
# they skip rather than fail on the Voyage path;
# tests/test_voyage_client.py covers the other backend.
local_only = pytest.mark.skipif(
    resolve_reranker_provider() != "local",
    reason="RERANKER_PROVIDER is not 'local'; no in-process cross-encoder is loaded",
)


@local_only
def test_reranker_model_identity():
    from retriever.rag_retriever import reranker
    assert reranker is not None
    assert reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"


@local_only
def test_reranker_orders_relevant_above_irrelevant():
    from retriever.rag_retriever import reranker

    query = "When was The Legend of Zelda: Breath of the Wild released?"
    relevant = "The Legend of Zelda: Breath of the Wild was released on March 3, 2017 for the Nintendo Switch."
    irrelevant = "The recipe calls for two cups of flour and a teaspoon of baking soda."

    scores = list(reranker.rerank(query, [irrelevant, relevant]))
    assert scores[1] > scores[0]
