"""
tests/test_llm_config.py

Unit tests for LLM configuration consistency (Gemini primary, Groq
fallback). Fully local — no network calls, no API keys required.
"""

import pathlib
import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_gemini_model_default():
    from llm.gemini_client import GEMINI_MODEL
    assert GEMINI_MODEL == "gemini-flash-latest"


def test_gemini_embed_model_and_dim():
    from llm.gemini_client import GEMINI_EMBED_MODEL, GEMINI_EMBED_DIM
    assert GEMINI_EMBED_MODEL == "gemini-embedding-001"
    assert GEMINI_EMBED_DIM == 768


def test_pricing_has_gemini_row():
    from llm.pricing import MODEL_PRICING
    assert "gemini-flash-latest" in MODEL_PRICING
    assert MODEL_PRICING["gemini-flash-latest"] == (0.0, 0.0)


def test_requirements_has_openai_no_modal():
    src = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "openai" in src
    assert "\nmodal" not in src and not src.startswith("modal")


def test_modal_service_files_removed():
    for path in ("llm/modal_llm.py", "llm/modal_embed.py", "llm/modal_rerank.py"):
        assert not (REPO_ROOT / path).exists(), f"{path} should have been deleted"


def test_reranker_model_matches_calibration():
    """retriever/quality_gate.py's REFUSE_FLOOR/WEAK_FLOOR are calibrated on
    Xenova/ms-marco-MiniLM-L-6-v2's raw logits — the in-process reranker
    must use that exact model, not a substitute."""
    from retriever.rag_retriever import reranker
    assert reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"
