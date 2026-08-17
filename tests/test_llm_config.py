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
    """quality_gate's _FLOORS["local"] is calibrated on
    Xenova/ms-marco-MiniLM-L-6-v2's raw logits — the in-process reranker
    must use that exact model, not a substitute. Only applies on the
    local path; under RERANKER_PROVIDER=voyage the ONNX model is
    deliberately never constructed."""
    from retriever.reranker_provider import resolve_reranker_provider

    if resolve_reranker_provider() != "local":
        pytest.skip("RERANKER_PROVIDER is not 'local'")

    from retriever.rag_retriever import reranker
    assert reranker is not None
    assert reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_reranker_provider_defaults_to_local(monkeypatch):
    """The flag must default to the calibrated in-process path, so an
    unset RERANKER_PROVIDER never silently switches score scales."""
    from retriever.reranker_provider import DEFAULT_PROVIDER, resolve_reranker_provider

    assert DEFAULT_PROVIDER == "local"

    monkeypatch.delenv("RERANKER_PROVIDER", raising=False)
    assert resolve_reranker_provider() == "local"

    # An unrecognized value falls back rather than raising — a typo'd env
    # var must not take the service down at import time.
    monkeypatch.setenv("RERANKER_PROVIDER", "vooyage")
    assert resolve_reranker_provider() == "local"

    monkeypatch.setenv("RERANKER_PROVIDER", ' "VOYAGE" ')
    assert resolve_reranker_provider() == "voyage"


def test_voyage_floors_are_uncalibrated_placeholder():
    """Voyage returns normalized 0..1 scores; until
    evaluation/calibrate_relevance.py has been re-run against the
    migrated corpus, its floors must stay None so the gate skips the
    ladder instead of thresholding a foreign scale."""
    from retriever.quality_gate import RetrievalQualityGate

    assert RetrievalQualityGate._FLOORS["local"] == (-3.0, 2.0)
    assert RetrievalQualityGate._FLOORS["voyage"] is None


def test_cloudflare_floors_are_uncalibrated_placeholder():
    """bge-reranker-base emits raw logits of a similar magnitude to
    ms-marco's, which makes borrowing local's floors look reasonable and
    be wrong. It must stay None until it has its own calibration run."""
    from retriever.quality_gate import RetrievalQualityGate

    assert RetrievalQualityGate._FLOORS["cloudflare"] is None


def test_hfspace_shares_local_floors():
    """The HF Space runs the same model as the in-process path, so it
    inherits local's calibration by design. If these ever diverge, the
    Space's model or pinned fastembed version has drifted from
    hf_space/requirements.txt and the entry must go back to None."""
    from retriever.quality_gate import RetrievalQualityGate

    assert (
        RetrievalQualityGate._FLOORS["hfspace"]
        == RetrievalQualityGate._FLOORS["local"]
    )


def test_hf_space_pins_the_calibrated_model_and_fastembed():
    """Score parity with the calibrated in-process path is the entire
    reason the Space exists — the model name must match, and fastembed
    must be pinned rather than floating."""
    space_app = (REPO_ROOT / "hf_space" / "app.py").read_text(encoding="utf-8")
    space_reqs = (REPO_ROOT / "hf_space" / "requirements.txt").read_text(encoding="utf-8")

    assert 'MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"' in space_app
    assert "fastembed==" in space_reqs
