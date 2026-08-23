"""
tests/test_llm_config.py

Unit tests for LLM configuration consistency (Gemini primary, Groq
fallback). Fully local — no network calls, no API keys required.
"""

import json
import pathlib
import re
from datetime import date

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_gemini_model_default():
    """Defaults to a lite model on purpose: gemini-flash-latest resolves
    to a 20-request/day model that closes long streams mid-answer."""
    from llm.gemini_client import GEMINI_MODEL
    assert GEMINI_MODEL == "gemini-flash-lite-latest"


def test_gemini_embed_model_and_dim():
    from llm.gemini_client import GEMINI_EMBED_MODEL, GEMINI_EMBED_DIM
    assert GEMINI_EMBED_MODEL == "gemini-embedding-001"
    assert GEMINI_EMBED_DIM == 768


def test_pricing_has_gemini_row():
    from llm.gemini_client import GEMINI_MODEL
    from llm.pricing import MODEL_PRICING
    assert "gemini-flash-latest" in MODEL_PRICING
    assert MODEL_PRICING["gemini-flash-latest"] == (0.0, 0.0)
    # Whichever Gemini model is actually in use must be priced, or its
    # 0.0 cost is a missing row rather than a free tier.
    assert MODEL_PRICING.get(GEMINI_MODEL) == (0.0, 0.0)


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
    deliberately never constructed.

    Both the skip check and the assertion below read
    retriever.rag_retriever's own frozen RERANKER_PROVIDER/reranker --
    not a fresh resolve_reranker_provider() call. That module resolves
    the provider once at import time (deliberately, per its own
    comment), so whichever test in the session imports it first pins
    `reranker` for the rest of the process; re-resolving the env var
    live here could disagree with that frozen state and make this test
    contradict its own skip condition depending on suite order."""
    from retriever import rag_retriever

    if rag_retriever.RERANKER_PROVIDER != "local":
        pytest.skip("RERANKER_PROVIDER is not 'local'")

    assert rag_retriever.reranker is not None
    assert rag_retriever.reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"


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


def test_cloudflare_floors_are_calibrated():
    """bge-reranker-base emits a normalized 0..1 score, a different scale
    from ms-marco's raw logits — its floors must be its own calibration
    (evaluation/calibrate_relevance.py against the fully-migrated corpus),
    never borrowed from "local", and must sit inside the 0..1 range that
    scale actually produces."""
    from retriever.quality_gate import RetrievalQualityGate

    floors = RetrievalQualityGate._FLOORS["cloudflare"]
    assert floors is not None
    refuse_floor, weak_floor = floors
    assert refuse_floor < weak_floor
    assert 0.0 <= refuse_floor <= 1.0
    assert 0.0 <= weak_floor <= 1.0


def test_active_provider_floors_are_calibrated():
    """T24/§23: the honesty gate must not be silently switched off, or
    silently thresholding a STALE calibration, for whichever provider
    is actually running.

    Checks every provider with non-None floors against the calibration
    artifact RetrievalQualityGate._CALIBRATION records for it, rather
    than only asserting "is not None" — a stale artifact (measuring a
    corpus that no longer exists post-Gemini-embedding-migration) is
    not None either, and used to pass this test while describing a
    distribution that isn't there anymore. That was T24's bug.
    Deliberately red if a provider's artifact predates
    RetrievalQualityGate.CORPUS_EMBEDDING_MIGRATION_DATE, or if its
    floors contradict what its own artifact measured — those failures
    ARE the signal this test exists to raise. Still deliberately red
    under RERANKER_PROVIDER=voyage (still uncalibrated)."""
    from retriever.quality_gate import RetrievalQualityGate
    from retriever.reranker_provider import resolve_reranker_provider, VALID_PROVIDERS

    active = resolve_reranker_provider()
    assert RetrievalQualityGate._FLOORS[active] is not None, (
        f"RERANKER_PROVIDER={active!r} has no calibrated relevance floors — "
        "the honesty gate silently no-ops for every request. Run "
        "evaluation/calibrate_relevance.py and set _FLOORS accordingly."
    )

    results_dir = REPO_ROOT / "evaluation" / "results"

    for provider in VALID_PROVIDERS:
        floors = RetrievalQualityGate._FLOORS[provider]
        if floors is None:
            continue
        refuse_floor, weak_floor = floors

        artifact_name = RetrievalQualityGate._CALIBRATION.get(provider)
        assert artifact_name, (
            f"_FLOORS[{provider!r}] = {floors!r} but _CALIBRATION[{provider!r}] "
            "is unset — floors with no recorded artifact cannot be verified as current."
        )
        artifact_path = results_dir / artifact_name
        assert artifact_path.exists(), (
            f"_CALIBRATION[{provider!r}] names {artifact_name!r}, which does not "
            f"exist under {results_dir} — floors point at a missing calibration run."
        )

        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        generated = date.fromisoformat(artifact["generated"])
        assert generated >= RetrievalQualityGate.CORPUS_EMBEDDING_MIGRATION_DATE, (
            f"{artifact_name} was generated {generated}, before the corpus embedding "
            f"migration ({RetrievalQualityGate.CORPUS_EMBEDDING_MIGRATION_DATE}) — it "
            "measured a different corpus and no longer describes what the reranker "
            "scores today. Re-run evaluation/calibrate_relevance.py."
        )

        # The artifact must have been produced by this provider itself,
        # or by whichever provider it shares an artifact with (e.g.
        # hfspace -> local) — resolved via _CALIBRATION identity rather
        # than a hardcoded pair, so a future shared-model provider
        # needs no new special case here.
        owner = next(
            p for p, name in RetrievalQualityGate._CALIBRATION.items()
            if name == artifact_name
        )
        assert artifact["reranker_provider"] in (provider, owner), (
            f"{artifact_name} was generated for provider "
            f"{artifact['reranker_provider']!r}, not {provider!r} or its shared "
            f"owner {owner!r}."
        )

        answerable = artifact["answerable_relevance"]
        assert refuse_floor < answerable["min"], (
            f"_FLOORS[{provider!r}].refuse_floor={refuse_floor} is not strictly "
            f"below {artifact_name}'s answerable minimum ({answerable['min']}) — "
            "it would false-refuse evidence its own calibration run scored as answerable."
        )
        assert refuse_floor < weak_floor <= answerable["max"], (
            f"_FLOORS[{provider!r}]=({refuse_floor}, {weak_floor}) is inconsistent "
            f"with {artifact_name}'s answerable range (min={answerable['min']}, "
            f"max={answerable['max']})."
        )


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


def test_root_fastembed_pin_matches_hf_space():
    """quality_gate's _FLOORS["hfspace"] == _FLOORS["local"] rests on both
    services running the identical fastembed build. That invariant is
    only enforceable if both requirements files pin the same version --
    a floating root pin would let the two drift apart silently."""
    root_reqs = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    space_reqs = (REPO_ROOT / "hf_space" / "requirements.txt").read_text(encoding="utf-8")

    root_match = re.search(r"^fastembed==(\S+)", root_reqs, re.MULTILINE)
    space_match = re.search(r"^fastembed==(\S+)", space_reqs, re.MULTILINE)

    assert root_match, "requirements.txt must pin fastembed==<version>, not float it"
    assert space_match, "hf_space/requirements.txt must pin fastembed==<version>"
    assert root_match.group(1) == space_match.group(1)


def test_requirements_header_names_live_models():
    """The header comment is documentation, not code -- nothing forces it
    to track a model swap. Assert it names the live defaults so a future
    swap that forgets the header turns the build red instead of just
    quietly lying to the next reader."""
    from llm.gemini_client import GEMINI_MODEL
    from llm.ragent_client import _GROQ_MODEL

    src = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    header = src.split("# ----------", 1)[0]

    assert GEMINI_MODEL in header
    assert _GROQ_MODEL in header


def test_create_schema_dense_size_matches_gemini_dim():
    """DENSE_VECTOR_SIZE must track whatever Gemini actually returns, and
    the retired E5 model it was renamed away from must not resurface."""
    from llm.gemini_client import GEMINI_EMBED_DIM
    from vector.create_schema import DENSE_VECTOR_SIZE

    assert DENSE_VECTOR_SIZE == GEMINI_EMBED_DIM

    src = (REPO_ROOT / "vector" / "create_schema.py").read_text(encoding="utf-8")
    assert "E5" not in src
