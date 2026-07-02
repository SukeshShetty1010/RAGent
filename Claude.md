# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project summary
RAGent is a capability-aware agentic RAG system for gaming intelligence. It answers gaming queries using multi-source ingestion (RAWG, IGDB, GameSpot), hybrid Qdrant retrieval, an honesty gate, and Groq/Modal-hosted LLM infrastructure.

## What matters most
- Prefer evidence-backed answers over guesses.
- Preserve the backend/UI contract: the backend is the source of truth, and the UI must treat returned data as an immutable snapshot.
- Keep changes aligned with the existing deterministic routing, retrieval, and capability-assessment flow.
- Maintain graceful degradation in all error handling; never replace a safe partial response with a crash.

## Commands

```bash
# Activate env (always run project code inside this)
RAG_env\Scripts\activate            # Windows
source RAG_env/bin/activate         # Linux/macOS

# Install deps
pip install -r requirements.txt

# Ingest a single game (RAWG + IGDB + GameSpot -> Qdrant)
python -m upsert.upsert_all --game "Far Cry 5"

# Batch ingest
python -m scripts.bulk_ingest

# Run backend API (FastAPI + SSE streaming)
uvicorn api.main:app --port 8000

# Run frontend (Next.js)
cd frontend && npm install && npm run dev   # dev server
npm run build && npm run start              # production build
npm run lint                                # eslint

# Tests
python -m pytest tests/                     # full suite
python -m pytest tests/test_llm_config.py   # single file
python -m pytest tests/test_llm_config.py::test_name -v   # single test
python tests/regression_suite.py            # regression suite (not pytest-based)
python tests/verify_engine.py               # engine smoke/verification script

# KPI / evaluation dashboard
python -m KPI.Unified_KPI_Runner

# Modal (LLM + embedding services)
modal deploy llm/modal_llm.py
modal deploy llm/modal_embed.py
```

There is no lint/format command configured for the Python side (no ruff/black/flake8 config present) — match existing style by hand. `tests/` mixes real pytest tests (`test_*.py`) with standalone scripts (`regression_suite.py`, `verify_engine.py`, `KPI_run.py`, `evaluation_runner.py`) that are run directly, not via pytest.

## Architecture

7-step pipeline, orchestrated by `engine/execution_engine.py` (`RageEngine.run()`) and its streaming counterpart `engine/execution_engine_streaming.py` (`StreamingRageEngine.run_streaming()`, used by the SSE API):

1. **Intent extraction** — `agent/intent/intent_extractor.py`: regex-based `IntentSignalExtractor` detects signals like comparison/listicle.
2. **Task routing** — `agent/task_router.py`: `TaskRouter.route()` deterministically maps intent → `TaskType` / `RouterDecision`.
3. **Strategy selection** — `retriever/strategy_selector.py`: `StrategySelector.select()` picks a `RetrievalConfiguration` (e.g. decomposition for comparisons).
4. **Retrieval** — `retriever/orchestrator.py` (`RetrievalOrchestrator.run()`) drives `retriever/rag_retriever.py`'s hybrid Qdrant search (BM25 sparse + dense E5 vectors), gated by `retriever/quality_gate.py` (`RetrievalQualityGate` → `QualityReport` OK/WEAK/EMPTY). Weak/temporal results fall back to `agent/tools/web_search.py` (Tavily).
5. **Capability assessment** — `agent/capability/capability_assessor.py`: the honesty gate. `CapabilityAssessor.assess()` scores evidence sufficiency into `FULL`/`PARTIAL`/`INSUFFICIENT`; `INSUFFICIENT` bypasses the LLM entirely (safe refusal instead of a hallucinated answer).
6. **Context assembly** — `agent/context_assembler.py` + `agent/context_algorithms.py`: dedupes chunks and applies character/token budget under entity-balanced ordering.
7. **Prompt + generation** — `agent/prompt_manager.py` applies a 3-stage prompt fallback (verbose → concise → truncated) to stay within budget, then generates via `llm/ragent_client.py`/`ragent_client_streaming.py` (Groq `llama-3.1-8b-instant`, default) or `llm/modal_llm.py` (Modal-hosted `google/gemma-3-12b-it` on L40S via vLLM, optional).

Supporting layers:
- `ingest/`, `data/`, `pre_process/`, `upsert/`: multi-source ETL. `data/*.py` fetch raw API payloads; `pre_process/cleaner.py` and `merge.py` clean/normalize into canonical schema; `upsert/upsert_all.py` performs a 5-stage idempotent upsert (Game → Platform → IGDB → GameSpot → Editorial) into Qdrant. Game identity uses deterministic IDs (`unified_game_id = slug-year-sha1[:8]`) and a "No-Orphan Rule" — all entities must link to a canonical Game anchor.
- `vector/create_schema.py`: defines the 5 Qdrant collections (EditorialChunk, Game, PlatformSpec, IGDB_Game, GameSpot_Game).
- `chunking/editorial_chunker.py`: word-based chunker (500 tokens, 50 overlap) for GameSpot editorial content, embedded via `llm/modal_embed.py` (`E5Embedder`, `intfloat/e5-base-v2` on Modal T4 GPU).
- `api/`: FastAPI backend (`main.py`) exposing `/api/chat` as an SSE stream (token/stage/done/error events) plus `/health` and `/ping`. This is the current backend surface — `ui/app_streaming.py` (Streamlit) is a legacy/alternate frontend; `frontend/` (Next.js) is the primary UI and treats API responses as an immutable snapshot per the backend/UI contract.
- `utils/observability.py`: thread-safe `MetricsRegistry` / `ProfileBlock` for latency profiling used throughout the pipeline.
- `utils/caching.py`: deterministic semantic caching via `@cacheable(ttl_seconds)`.
- `KPI/Unified_KPI_Runner.py`: orchestrates 5 evaluation modules (grounding fidelity, honesty rate, routing accuracy, retrieval quality, latency attribution) into one dashboard.

## Architecture rules to respect
- `agent/`: intent detection, routing, capability assessment, context assembly, prompt management.
- `retriever/`: hybrid search, quality gating, strategy selection, evidence orchestration.
- `ingest/`, `data/`, `upsert/`: source ingestion and Qdrant upserts for RAWG, IGDB, and GameSpot.
- `llm/`: Modal-backed generation and embedding services, plus Groq streaming client.
- `api/` and `frontend/`: current backend/UI surface; the backend is the source of truth and the frontend must not recompute or re-derive business logic.

## Core operating rules
- Use deterministic routing when adding or modifying query handling.
- Keep the honesty gate intact: responses must be FULL, PARTIAL, or INSUFFICIENT based on evidence sufficiency.
- Preserve hybrid retrieval behavior: BM25 + vector search through Qdrant.
- Do not move business logic into the UI layer.
- Maintain type hints throughout Python code.
- Prefer soft fallbacks and explicit failure states over broad exceptions that hide errors.

## Environment rules
- Always use the `RAG_env` virtual environment to run project code.
  - Windows: `RAG_env\Scripts\activate`
  - Linux/macOS: `source RAG_env/bin/activate`
- Python version: 3.10+
- Required environment variables must exist in `.env` before running ingestion, retrieval, or LLM code.
- Be careful with Modal, Qdrant, RAWG, IGDB, GameSpot, and Tavily credentials; do not hardcode secrets.

## Working style
- Before making changes, inspect the relevant module(s) and follow existing conventions.
- Prefer small, local edits that preserve the current pipeline and naming.
- When adding logic, keep it testable and deterministic.
- When fixing bugs, identify the failing layer first: ingest, retrieval, capability assessment, context assembly, prompt construction, LLM, or UI.

## RAGent-specific implementation reminders
- Retrieval quality and capability assessment are central; do not weaken them for convenience.
- Keep context assembly compact and relevant.
- Respect prompt budget constraints.
- Maintain observability and deterministic latency/profiling behavior.
- When the evidence is weak, degrade safely instead of hallucinating.

## If working on ingestion
- Preserve idempotent upserts and canonical game identity handling.
- Keep the multi-source ETL pipeline compatible with RAWG, IGDB, and GameSpot.
- Clean and normalize data before upserting.

## If working on UI
- Treat backend output as immutable.
- Do not recalculate metrics or re-derive business logic in the frontend.
- Render the execution result structure as provided.

## If working on LLM / Modal
- Keep Modal client usage compatible with the current deployment pattern.
- Preserve any lazy initialization / environment loading patterns already in place.
- Do not break local development or CI compatibility.
- When writing Modal library-related code, always use `@llm.txt` to refer to the Modal library documentation.

## If working on evaluation / KPI code
- Keep metrics deterministic and reproducible.
- Preserve existing evaluation outputs and latency attribution logic.
- Prefer additive changes over redesigns.

## Remember
This repo is optimized for honest, evidence-backed answers. When uncertain, prefer explicit refusal or partial grounding over confident speculation.
