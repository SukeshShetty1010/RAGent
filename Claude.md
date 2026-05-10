# RAGent — Claude Project Instructions

This file gives Claude the minimum persistent context needed to work safely and effectively in this repository.

## Project summary
RAGent is a capability-aware agentic RAG system for gaming intelligence. It answers gaming queries using multi-source ingestion, hybrid retrieval, an honesty gate, and Modal-hosted LLM infrastructure.

## What matters most
- Prefer evidence-backed answers over guesses.
- Preserve the backend/UI contract: the backend is the source of truth, and the UI must treat returned data as an immutable snapshot.
- Keep changes aligned with the existing deterministic routing, retrieval, and capability-assessment flow.
- Maintain graceful degradation in all error handling; never replace a safe partial response with a crash.

## Architecture to respect
- `agent/`: intent detection, routing, capability assessment, context assembly, prompt management.
- `retriever/`: hybrid search, quality gating, strategy selection, evidence orchestration.
- `ingest/`, `data/`, `upsert/`: source ingestion and Qdrant upserts for RAWG, IGDB, and GameSpot.
- `llm/`: Modal-backed generation and embedding services.
- `ui/`: Streamlit UI that follows the immutable UI contract.
- `scripts/` and `KPI/`: batch jobs, evaluation, observability, and reporting.

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
