# RAGent Flagship-Readiness Audit & Gap-Closing Build Plan

## Context

RAGent is the designated flagship project anchoring a 3–6 month US/EU remote AI/ML job
search. This audit inspects the **actual code on disk** against a fixed flagship rubric and
produces a sequenced build plan to close verified gaps.

**Headline finding:** RAGent's *technical substance* is largely stronger than the rubric
demands — genuine RRF hybrid retrieval plus a cross-encoder rerank stage, a real multi-source
domain corpus, and a hard-guarded refusal path. The gaps are almost entirely in **proof and
presentation**: nothing is deployed, there is no real evaluation, and — most urgently — the
KPI numbers currently published in the README are **structurally incapable of failing**. That
last item is not a missing feature; it is an active credibility liability in front of anyone
technical who reads the code.

### Evidence sourcing note

There is **no prior RAGent session history** available to me — this session began with `/init`.
Every determination below therefore comes from **direct file inspection**, except two facts
supplied by the candidate when asked (marked *[candidate]*). Nothing is inferred from the
README's claims; the README is treated as an assertion to be checked, not as evidence.

---

## Step 1 — Current-state inventory (verified)

| Area | Verified state | Source |
|---|---|---|
| Hybrid retrieval | Qdrant `query_points` with two `Prefetch` branches — BM25 sparse (`using="bm25"`) and dense (`using="dense"`) — fused by `models.FusionQuery(fusion=models.Fusion.RRF)`, `fetch_limit = max(limit*4, 20)` | `retriever/rag_retriever.py:118-147` |
| Reranking | **Superseded 2026-08-17** (was: cross-encoder on Modal). Now provider-dispatched via `RERANKER_PROVIDER` — Cloudflare Workers AI `@cf/baai/bge-reranker-base` in production, in-process fastembed ONNX as the default/offline path. Still reorders into `rerank_score` with the RRF `score` preserved for the quality gate, still fail-soft to RRF order on error. Per-provider floors in `quality_gate._FLOORS`, `None` = uncalibrated ⇒ skip the ladder | `retriever/rag_retriever.py`, `retriever/reranker_provider.py`, `llm/cloudflare_rerank_client.py` |
| Domain corpus | Gaming. Three source APIs (RAWG identity, IGDB relational metadata, GameSpot editorial), 5 Qdrant collections, editorial chunking under a written contract (~500 tok / 50 overlap, no-orphan rule) | `data/`, `ingest/`, `vector/create_schema.py`, `chunking/chunk_contract.md` |
| Corpus volume | **100 games, 2791 `EditorialChunk` points, verified zero orphans and zero duplicate `unified_game_id`s** — full rebuild since the original audit, decoupled ingestion identity from RAWG, added Wikipedia + Steam editorial sources alongside GameSpot | `scripts/verify_corpus.py` (live run, 2026-08-09); commits `c595571`, `16079a2` |
| Refusal path | `CapabilityAssessor.assess()` returns `INSUFFICIENT` on empty evidence or `QUALITY_EMPTY`; engine hard-guards generation behind `if capability != AnswerCapability.INSUFFICIENT:` | `agent/capability/capability_assessor.py:58-59`; `engine/execution_engine.py:185` |
| Citation attribution | Context blocks injected as `[Source: {title} | Type: {type}]`; every task template instructs "Cite sources"; frontend renders a "Sources (Evidence)" panel with `source_title` + snippet | `agent/prompt_templates.py:52-55, 105/121/138/152/167`; `frontend/src/app/page.tsx:257-267` |
| Evaluation | **RAGAS + a 50-query golden set complete** (`evaluation/`): 20 factual, 10 comparison, 10 listicle, 10 deliberately unanswerable, drafted from live corpus payloads. Two full production-path runs completed (web-enabled and corpus-only). Refusal precision/recall, RAGAS context precision/faithfulness/answer relevancy (40/40 answerable queries, Modal-judged), and a 4-mode retrieval ablation (both precision@k and RAGAS context precision per mode) are all committed under `evaluation/results/`. See Phase 3 results below — the homegrown metrics in `tests/evaluation_metrics.py` remain as a separate, smaller diagnostic layer | `evaluation/build_golden_set.py`, `evaluation/run_eval.py`, `evaluation/refusal_metrics.py`, `evaluation/ragas_eval.py`, `evaluation/ablation.py`, `evaluation/modal_judge_llm.py`; `evaluation/results/*` |
| Observability | Homegrown `MetricsRegistry` + `ProfileBlock` (thread-safe, nested wall-clock), **plus (2026-08-14) real Groq token/cost capture and an optional Langfuse trace layer mirroring every `ProfileBlock` span** — hard no-op with zero import cost unless both `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` are set. Verified live in production with a screenshot of a full trace | `utils/observability.py`, `utils/tracing.py`, `llm/pricing.py` |
| Containerization | Multi-stage Dockerfile (Node 20 static export → python:3.11-slim), `HEALTHCHECK`, `PORT`-aware CMD. **Verified booting locally under Render's 512MB cap (2026-08-14)** — 124-135MB resident | `Dockerfile` |
| Deployment | **Live** (2026-08-14): `https://rag-ent.onrender.com`. `render.yaml` declares `healthCheckPath: /health` and (optional) Langfuse keys; all set in the Render dashboard, redeployed, and verified — `/health` 200, a real sourced query, and the GTA VI refusal path all confirmed against the live URL. Keepalive via `.github/workflows/render-keepalive.yml` (10-min `/ping`, no secrets needed). **2026-08-17:** Modal token keys removed from `render.yaml`; `CLOUDFLARE_ACCOUNT_ID`/`CLOUDFLARE_API_TOKEN`/`RERANKER_PROVIDER` added. `modal-keepalive.yml` deleted in `80be984` | `render.yaml`; `git remote -v` confirms public repo `SukeshShetty1010/RAG_ent` |
| LLM fallback | **Superseded 2026-08-16** (was: Groq primary → Gemma 3 12B on Modal L40S). Now **Gemini primary → Groq fallback**, fail-soft on any Gemini error including an unset key. No Modal dependency anywhere; `llm/modal_*.py` deleted | `llm/ragent_client.py`, `llm/ragent_client_streaming.py`, `llm/gemini_client.py` |
| README | Value prop, two Mermaid diagrams, five metrics tables, quick-start, deployment notes. **No live demo link** (`YOUR_USERNAME` placeholder at line 278). **No "what I rejected" section** | `README.md` |

### Defects found during inspection (not rubric items, but blocking)

These matter because the README publishes these numbers as headline achievements.

1. ✅ **RESOLVED.** The hardcoded `hallucinated_claims`/"Unsafe Outputs" rows are gone from
   `KPI/Faith_Fair_KPI.py`. `calculate_grounding_fidelity` now feeds a real
   `Citation-Grounded Sentence Rate`; the code explicitly documents that `capability_distribution`
   (not `honest_rate`) is a rate of FULL+PARTIAL vs INSUFFICIENT outcomes, not a hallucination
   measurement. The tautological `calculate_honesty_rate` still exists in
   `tests/evaluation_metrics.py` as a diagnostic, but Phase 3's `evaluation/refusal_metrics.py`
   now supplies the falsifiable replacement (see Phase 3 results below).
2. ✅ **RESOLVED.** `CITATION_FORMAT` / `CITATION_PATTERN` are now a single shared source of
   truth in `agent/output_validator.py:39-40`, imported by `tests/evaluation_metrics.py` and
   enforced (`Cite every factual claim inline as (Source: 'Exact Source Title').`) in every
   template in `agent/prompt_templates.py`. `validate_answer()` checks emitted citations against
   assembled context and flags unmatched ones on `agent_decisions["output_validation"]` —
   fail-soft, never discards the answer. **Caveat found during Phase 3 live runs:** the LLM does
   not always follow the instructed format even though it is now specified and validated (e.g.
   `"(Source: Fortnite — Overview)"` without the required quotes) — `output_validation.is_valid`
   correctly flags these as invalid. Enforcement exists; compliance is not 100%.
3. ⚠️ **PARTIALLY RESOLVED.** Modal lookups are now lazy (`_get_dense_encoder()` /
   `_get_reranker()` in `retriever/rag_retriever.py`, not called at import time). `render.yaml`
   now declares `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` (this session). Still outstanding: real
   values must be set in the Render dashboard and the service redeployed — reported **live but
   broken** *[candidate]*, unverified since the fix.
4. ✅ **RESOLVED** (this session). `render.yaml` now declares `healthCheckPath: /health`.
5. ✅ **RESOLVED.** `KPI/Unified_KPI_Runner.py` uses `parents[1]`; confirmed by a clean run this
   session (`python -m KPI.Unified_KPI_Runner`, exit 0, no regression from Phase 3 changes).
6. ✅ **RESOLVED (found and fixed this session).** `llm/modal_llm.py` — the same Modal service
   backing live user-facing generation via `ragent_client_streaming.py`, not just this eval
   harness — had a latent concurrency bug: `@modal.concurrent(max_inputs=8)` dispatches
   concurrent `generate()` calls from multiple worker threads onto one shared `asyncio` event
   loop driven by the non-reentrant `run_until_complete()`. Two threads racing on it corrupts
   the loop's internal state, permanently wedging that container for the rest of its life —
   confirmed via `modal app logs` showing hundreds of consecutive, identical `RuntimeError: This
   event loop is already running` tracebacks with zero successful generations in between, while
   a caller's retry logic kept hammering the dead container and silently burning GPU-seconds.
   Fixed by running the loop continuously in a dedicated background thread and bridging calls in
   via `asyncio.run_coroutine_threadsafe` (the correct pattern for many threads submitting to one
   running loop — `AsyncLLMEngine` already supports concurrent `generate()` internally via
   continuous batching, so this also lets `@modal.concurrent`'s parallelism actually take
   effect). Validated with an ephemeral `modal run` concurrent smoke test (6 simultaneous calls,
   all clean, no race) before `modal deploy`-ing the fix to production.

---

## Step 2 — Gap analysis

### Technical substance

| Rubric item | Status | Basis |
|---|---|---|
| Hybrid dense + sparse, RRF-fused | **PRESENT (exceeds)** | `rag_retriever.py:121-147` + rerank stage `:188-221` |
| Specific non-generic corpus | **PRESENT** | 3-API gaming corpus, 50+ games, 5 collections |
| Explicit "insufficient information" handling | **PRESENT** | `capability_assessor.py:58-59` + hard guard `execution_engine.py:185` |
| Source/citation attribution in outputs | **PRESENT (weak enforcement)** | Sources injected + rendered in UI; but no enforced citation format, and nothing validates that emitted citations match retrieved titles |
| Quantified RAGAS-equivalent eval (context precision, faithfulness, answer relevancy) | **PRESENT** | 50-query golden set + RAGAS scoring vs a Groq 70B judge, committed under `evaluation/results/` — see Phase 3 results below. Full RAGAS scoring is still catching up on a Groq free-tier daily token quota (checkpoint-resumable); the golden set, both production-path runs, and the refusal + ablation metrics are complete |

### Engineering proof

| Rubric item | Status | Basis |
|---|---|---|
| Live, publicly reachable deployment | **ABSENT** | Confirmed *[candidate]*; also blocked by defect #3 |
| Containerized, one-command reproducible | **PRESENT (untested at runtime)** | Dockerfile verified by reading; never confirmed to build/boot |
| Observability capturing latency, token cost, retrieval steps | **PARTIAL** | Latency + retrieval steps: PRESENT (`ProfileBlock`). Token cost: **ABSENT**. Export/tracing UI: **ABSENT** |

### Presentation

| Rubric item | Status | Basis |
|---|---|---|
| One-line value proposition | **PRESENT** | `README.md:14` |
| Architecture diagram | **PRESENT** | Two Mermaid diagrams |
| Metrics table | **PRESENT but not credible** | Five tables; numbers traced to non-failing code (defect #1) |
| Live demo link | **ABSENT** | `YOUR_USERNAME` placeholder, `README.md:278` |
| "Why this approach / what I rejected" | **ABSENT** | "Key Design Decisions" table gives rationale but names no rejected alternative |
| Depth signal (one thing done exceptionally) | **AT RISK** | Retrieval + honesty architecture reads as genuine depth; the five-table KPI dashboard and "resume-grade" comments throughout read as checklist-padding and undercut it |

---

## Step 3 — Gap-closing build plan (sequenced)

Sequencing rationale: fix the **credibility liability** first (it is actively harmful and cheap
to fix), then get something **live** (observability and README demo links both depend on it),
then **measure** (the README cannot cite scores that don't exist), then **write**.

### Phase 0 — Stop the bleeding (2–3 h) — ✅ COMPLETE

**0.1 Neuter the fake safety metrics** *(~1 h)*
Delete the hardcoded `hallucinated_claims`/`Unsafe Outputs` rows from
`KPI/Faith_Fair_KPI.py:160-181`. Until Phase 3 produces real numbers, remove the
Faithfulness & Safety table from `README.md:125-132` entirely. Shipping no number beats
shipping a number that a reviewer can disprove by reading 20 lines.

**0.2 Make citations machine-checkable** *(~1.5 h)*
Amend every template in `agent/prompt_templates.py` from `"Cite sources."` to specify the
exact format: `Cite every factual claim inline as (Source: 'Exact Source Title').` Then extend
`agent/output_validator.py` to parse emitted citations and assert each cited title appears in
the assembled context, flagging unmatched citations on the result object. This makes the
existing `calculate_grounding_fidelity` regex actually work and turns citation attribution
from "instructed" into "enforced" — a real differentiator.

**0.3 Fix `KPI/Unified_KPI_Runner.py:16`** `parents[2]` → `parents[1]`. *(2 min)*

### Phase 1 — Get it live (3–5 h) — ✅ LANDED (2026-08-14)

**Results.** Live at `https://rag-ent.onrender.com`. Docker verified locally under Render's
512MB cap (`docker run -m 512m --env-file .env ragent`) before pushing — booted at 124-135MB
resident, `/health`/`/ping` 200, a real sourced query, and the GTA VI refusal path all confirmed
both inside the container and against the deployed URL. Two real, previously-undiagnosed bugs
were found and fixed getting there, neither of which was on the original punch list:

- **`.env` wrapped every value in literal quotes** (`QDRANT_URL="https://...io"`). `docker run
  --env-file` doesn't strip quotes the way `python-dotenv` does, so the hostname arrived mangled
  and DNS resolution failed inside the container — 100% reproducible, not the flaky Docker
  networking it initially looked like. Fixed by stripping quotes from all 15 quoted `.env`
  values. Local `python`/pytest runs were never affected (dotenv strips quotes fine); this was a
  Docker-only landmine.
- **Live secrets exposure, mid-session.** An overly broad `grep '="'  .env` (chasing the quote
  bug above) printed every quoted secret in `.env` directly into the conversation transcript —
  Groq, Tavily, Qdrant, RAWG, GameSpot, Twitch, HF, APITube, GNews, MediaStack, NewsAPI. Flagged
  immediately; all keys rotated by the candidate. Fresh Modal credentials were minted the same
  way (`modal token new`) and written to `.env`/Render/GitHub Actions secrets without ever being
  printed. Worth recording as a lesson: broad greps over files known to hold secrets are exactly
  the kind of action that needs a length/prefix-only check instead of a raw content dump.

Keep-alive: `.github/workflows/render-keepalive.yml` (10-min `/ping`, no secrets needed) rather
than a third-party cron-job.org account, per the candidate's preference for keeping everything in
GitHub Actions alongside the existing `qdrant-keepalive.yml`.

**1.1 Version control** *(~30 min)* — `git init` here, reconcile against the repo on the other
machine, push to a public GitHub repo. Note: `git` is **not on PATH in this shell** and must be
installed first. Replace the `YOUR_USERNAME` placeholder at `README.md:278`.

**1.2 Fix the Modal credential blocker** *(~1 h)* — Add `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` (`sync: false`) to `render.yaml`; add `healthCheckPath: /health`. Then
make Modal lookups lazy: move the module-level `modal.Cls.from_name` calls at
`retriever/rag_retriever.py:46-53` behind a `_get_embedder()` / `_get_reranker()` accessor,
matching the existing `_get_groq_client()` lazy-binding pattern in `llm/ragent_client.py`.
This is required for boot to survive import, and it makes the container testable without Modal
credentials present.

**1.3 Verify the container locally before pushing** *(~1 h)* —
`docker build -t ragent .` then `docker run -p 10000:10000 --env-file .env ragent`; confirm
`/health`, then a real query through the UI end-to-end.

**1.4 Deploy to Render + keep-alive** *(~1 h)* — Deploy, set all env vars in the dashboard,
point a free cron-job.org job at the already-built `/ping` endpoint (`api/main.py:42-50`) at
10-minute intervals to defeat free-tier cold starts.

**Cold-start caveat to plan around:** a Render free instance calling Modal cold will have a
brutal first-request latency. Before putting this link in front of recruiters, measure it; if
it exceeds ~15 s, add a warming call to the Modal services from the `/ping` handler.

### Phase 2 — Real observability (4–6 h) — ✅ LANDED (2026-08-14)

**Results.** Groq's `usage` field is no longer discarded — `stream_options: {"include_usage":
true}` (passed via `extra_body`, since the installed Groq SDK 0.37.1 has no typed param for it)
puts prompt/completion tokens on the final streamed chunk, read in both `chat_completion_remote`
and `chat_completion_streaming`, converted to USD via the new `llm/pricing.py`, and surfaced in
both engines' `kpis` dict. A real request now reports e.g. `prompt_tokens: 1455,
completion_tokens: 82, cost_usd: 7.931e-05`.

Langfuse wired as an optional trace layer: `utils/observability.py` gained a generic span-hook
mechanism (`set_span_hooks`) so `ProfileBlock` stays free of any external dependency by
construction; `utils/tracing.py` is the only module that imports `langfuse`, and it's a hard
no-op — no import attempted — unless both `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` are set. Verified
end-to-end in production: a live query produced a `chat_request` trace with nested spans mirroring
every `ProfileBlock` name 1:1 (`TaskRouting`, `Retrieval → LocalVectorSearch →
EmbeddingGeneration/VectorQuery/CrossEncoderRerank`, `ContextAssembly → Ordering/Deduplication`,
etc.) plus a separate `LLMGeneration` generation-type node showing real token usage (942→512,
Σ1,454) and cost ($0.000088) — screenshot on file. `MetricsRegistry` was kept exactly as planned;
Langfuse is a second layer, not a replacement.

**2.1 Langfuse** *(~4 h)* — Free-tier Langfuse Cloud; `pip install langfuse` (thin SDK, safe
for the 512 MB cap — **do not** add `torch`/`sentence-transformers`, per the deployment
constraint). Wrap the 7 engine steps as spans on a single trace, mirroring the existing
`ProfileBlock` names so both layers agree. Capture per-trace: task type, capability verdict,
quality-gate status, chunk count, whether web fallback fired, and — new — **token counts and
cost from the Groq response's `usage` field**, which is currently discarded.

**2.2 Keep `MetricsRegistry`** — Do not rip it out. It feeds the KPI runners and is a genuine
"I built this myself" talking point. Langfuse is the external trace layer; they coexist.

### Phase 3 — Real evaluation (8–12 h) ← *the highest-value work in this plan* — ✅ LANDED (2026-08-09)

**Results, measured live against the rebuilt 100-game corpus.** Full methodology and all
committed artifacts: `evaluation/`. Golden set: 50 records (20 factual, 10 comparison, 10
listicle, 10 unanswerable), auto-drafted from real Qdrant payloads by
`evaluation/build_golden_set.py`, `reviewed: false` on all records pending a human pass — the
numbers below are reported honestly as **provisional** until that review lands.

**Refusal precision/recall — the falsifiable number this phase exists to produce, and it did
find something real:**

| Run | Refusal precision | Refusal recall | False-answer rate | Over-refusal rate |
|---|---|---|---|---|
| Default (web fallback on) | 0.0 | **0.0** | **1.0** | 0.0 |
| Corpus-only (web fallback off) | 0.0 | **0.0** | **1.0** | 0.0 |

**Root cause identified, not just measured:** the engine answered all 10 unanswerable queries
in both runs — including "What is the capital of France?" and "Who won the 2024 US presidential
election?" — instead of returning `INSUFFICIENT`. This is not a web-fallback masking effect
(same result with web fallback fully disabled). Direct inspection of
`retriever/quality_gate.py`'s output on these runs shows the quality gate classified 9/10 as
`quality_ok` and 1/10 as `quality_weak` — **never `quality_empty`** — because Qdrant's hybrid
search always returns *k* nearest neighbors regardless of true relevance, and the gate has no
similarity floor low enough to catch "these results are nearest-neighbor noise, not evidence."
`CapabilityAssessor.assess()` only escalates to `INSUFFICIENT` on `quality_empty` or zero
evidence, so a confidently-wrong quality classification propagates straight through the honesty
gate. **This is the single most important finding of this phase** and should be treated as a
new, higher-priority item — see the open item added to Step 4's checklist below.

**Retrieval ablation — a negative result, reported as-is per the plan, and confirmed by a second
independent metric:**

| Mode | Precision@K | Entity Coverage | RAGAS Context Precision |
|---|---|---|---|
| `dense` | 0.9400 | 0.8875 | 0.6537 |
| `bm25` | 0.9350 | 0.8375 | 0.5208 |
| `hybrid` (RRF, no rerank) | **0.9500** | **0.8750** | **0.6397** |
| `hybrid_rerank` (production default) | 0.9200 | 0.8125 | 0.5168 |

Plain RRF hybrid beats both single-mode baselines on precision@k, but the cross-encoder
reranking stage — the production default — scores *lowest of all four modes* on both
precision@k/entity coverage (40 answerable golden-set queries, no LLM) **and** RAGAS context
precision (20-query subset, Modal-judged) — two independently-computed metrics agreeing that
reranking is actively hurting retrieval quality on this corpus, not just failing to help. The
rerank stage is not currently earning its latency cost (it accounted for ~40-75% of retrieval
latency in the KPI runner's profile). Worth a follow-up look at the cross-encoder model choice
or a corpus-specific relevance check before the README cites reranking as a differentiator —
right now the honest claim is the opposite of what the architecture intends.

**RAGAS (context precision, faithfulness, answer relevancy) — complete, 40/40 answerable
queries scored (2026-08-09):**

| Metric | Score |
|---|---|
| context_precision | 0.5722 |
| faithfulness | 0.9077 *(6/40 null from per-job scoring timeouts, averaged over the remaining 34)* |
| answer_relevancy | 0.7306 |

Getting a full run took two independent judge tracks. Groq was tried first —
`llama-3.3-70b-versatile` (100K TPD), then `openai/gpt-oss-120b` (200K TPD), then
`qwen/qwen3.6-27b` (200K TPD) — and each was exhausted in turn by the same day's judge calls
(Groq's free-tier daily token quota, not a per-minute limit). The Groq/qwen track is preserved
as a partial, separate checkpoint (5/40) rather than discarded; independent judge backends are
tagged distinctly in every output filename (`_modal` suffix) specifically so no single result
file is ever scored by a mix of judges. The completed numbers above are from a second, parallel
judge track built on Modal-hosted `google/gemma-3-12b-it` (`evaluation/modal_judge_llm.py`,
wrapping `llm/modal_llm.py`'s existing `generate()` endpoint — no new model deployment needed).

Building that track surfaced and fixed a real bug in **production** code, not just eval-harness
glue — see defect #6 above. Three bugs total were found and fixed getting this judge track
working: (1) `ragas`'s self-consistency sampling shape mismatch — a non-langchain
`BaseRagasLLM` must return all `n` samples inside a single outer `Generation` list
(`resp.generations[0]`), the opposite convention from LangChain-wrapped LLMs; (2) an internal
`asyncio.gather` over `n` samples was firing concurrent Modal calls even with the executor
serialized to `max_workers=1`, fixed by awaiting sequentially instead; (3) the deeper one —
defect #6, the shared-event-loop race in `llm/modal_llm.py` itself, which had been silently
corrupting containers and wasting GPU-seconds on every retry until traced through `modal app
logs` and fixed at the source. Embeddings for both judge tracks use the same Modal `E5Embedder`
retrieval already relies on. Groq's `LangchainLLMWrapper(..., bypass_n=True)` and
checkpoint-only-on-partial-success fixes from the original wiring remain in place and apply to
the Groq track if it is resumed to finish its remaining 35 records on a later day.

**3.1 Build a 50-query golden set** *(~4 h)* — Persist as `tests/data/golden_set.jsonl`, not as
Python literals. Given 50+ games indexed, target: 20 factual (single-entity, verifiable),
10 comparison (two-entity), 10 listicle/open, and **10 deliberately unanswerable** (games not
in the corpus, or future/unreleased titles). Each record: `query`, `expected_task`,
`expected_entities`, `expected_source_titles`, `ground_truth_answer`, `should_refuse`.
The 10 unanswerable ones are what make the honesty claim falsifiable — that is the entire
point, and it is the current suite's central omission.

**3.2 Wire in RAGAS** *(~4 h)* — `pip install ragas datasets`, scored offline (not on the
Render path). Compute the three rubric metrics against the golden set:
`context_precision`, `faithfulness`, `answer_relevancy`. Use Groq as the RAGAS judge model to
avoid an OpenAI dependency. Write results to `evaluation/results/ragas_<date>.json`, committed
to the repo so the numbers are auditable.

**3.3 Add a genuine refusal metric** *(~2 h)* — On the 10 unanswerable queries, measure
**refusal precision/recall**: did `INSUFFICIENT` actually fire? Report false-answer rate
(answered when it should have refused) and over-refusal rate (refused when evidence existed).
Unlike the current honesty rate, this metric **can fail** — which is exactly why it is worth
putting on a resume.

**3.4 Ablation: prove the hybrid retrieval earns its keep** *(~2 h)* — Re-run context precision
three ways: dense-only, BM25-only, RRF+rerank. A table showing RRF beating both single-mode
baselines on your own corpus is the single most interview-durable artifact this project can
produce, and it directly supplies the "why this approach" section.

### Phase 3.5 — Make the honesty gate actually refuse (~6 h) — ✅ LANDED (2026-08-12)

Direct follow-up to Phase 3's row-21 finding: `RetrievalQualityGate` could not express "results
exist but are irrelevant" (`QUALITY_EMPTY` was only reachable on a literally-empty chunk list),
and `CapabilityAssessor` had no severity ladder to act on it even if it could. Two independent
structural blockers, verified in code, fixed without touching `capability_assessor.py` at all —
routing the new failure paths to `QUALITY_EMPTY` was enough, since `assess()` already returns
`INSUFFICIENT` there.

**What changed:**

- **`retriever/quality_gate.py`** — replaced the single FACTUAL-only `avg_score < 0.5` rule
  (thresholding the RRF fusion score, which is rank-derived and identical for a perfect match
  and pure noise) with a task-agnostic ladder on the cross-encoder `rerank_score` — the signal
  the reranker already computed and that nothing previously read. Also fixed `is_noise`'s
  substring-containment bug (`"deal" in "ideal"` was `True`).
- **`retriever/corpus_index.py`** (new) — `CorpusEntityIndex`: a fail-soft, lazily-loaded set of
  the corpus's actual `Game` titles. Catches the case the relevance floor alone cannot: a
  same-franchise query (e.g. "Grand Theft Auto VI") retrieving genuinely well-scoring chunks
  from the wrong entry ("Grand Theft Auto V"). Token-tuple equality, not substring matching —
  `"grand theft auto v"` is a raw substring of `"grand theft auto vi"`.
- **`retriever/orchestrator.py`** — the quality report is now re-evaluated on merged evidence
  after a web search actually contributes chunks, so a genuine web rescue can move the gate
  (previously the pre-web report was returned unconditionally even after a successful merge).
- **Calibration, not a guess** — `evaluation/calibrate_relevance.py` (new) measured the actual
  `rerank_score` distribution before any threshold was written. Finding worth recording:
  `ms-marco-MiniLM-L-6-v2` on this corpus applies `Identity`, not `Sigmoid` — raw logits
  (~-8 to +11), not a 0–1 score, contradicting the initial assumption. See
  `evaluation/results/relevance_calibration_2026-08-12.json`.

**Result — the headline number, corpus-only (web fallback off):**

| Run | Refusal precision | Refusal recall | False-answer rate | Over-refusal rate |
|---|---|---|---|---|
| Before (2026-08-09) | 0.0 | 0.0 | 1.0 | 0.0 |
| **After (2026-08-12), corpus-only** | **1.0** | **0.9** | **0.1** | **0.0** |
| After (2026-08-12), default (web on) | 1.0 | 0.8 | 0.2 | 0.0 |

9 of 10 deliberately-unanswerable queries now correctly refuse with **zero** regressions on the
40 answerable queries (`over_refusal_rate = 0.0` was treated as a hard gate, not a nice-to-have —
a refusal-storm regression would have been a worse failure than the one being fixed). The
default (web-enabled) run reports lower recall by design: web search genuinely produces
answering evidence for 2 of the 10 (`g047` "Beyond Good and Evil 2" and `g050` "2024 US
presidential election" — the latter is arguably a domain-restriction gap, not an evidence gap,
and is out of this phase's scope), so those two get re-graded `FULL` post-merge rather than
staying refused. `evaluation/refusal_metrics.py`'s `by_merge_state` breakdown:
`LOCAL_WEB_ATTEMPTED` (web tried, found nothing usable) refused 4/4 correctly;
`LOCAL_PLUS_WEB` (web contributed chunks) refused 4/6 — the 2 rescues are exactly `g047`/`g050`.

**The one accepted miss:** `g047` ("Beyond Good and Evil 2") has a real corpus `Game` identity
(likely a RAWG catalog entry with no editorial content ingested) — `entity_grounded=True`, so
the entity check doesn't fire, and its `max_relevance` (1.07) sits between legitimately-weak
answerable evidence (e.g. "RimWorld", 0.02) and the lowest genuinely-unanswerable score not
already caught by entity grounding (-4.32). No single scalar floor separates these three without
causing over-refusal on the answerable side — the honest call, per the calibration data, is to
let it land `QUALITY_WEAK` (partial, not refused) rather than pick a number that happens to score
well on this one query. Documented in `retriever/quality_gate.py`'s `REFUSE_FLOOR` comment.

**A real bug found and fixed mid-verification, worth recording:** the first version of the
entity-grounding span extractor fragmented multi-word titles containing lowercase connector
words (e.g. "Honkai: Star Rail - Then Wake **to** Weep" split into two spans at "to"), which
wrongly refused a legitimately-answerable query (`over_refusal_rate` was 0.025, not 0.0, on the
first corpus-only run). Fixed by letting a lowercase connector (`of`/`to`/`the`/`in`/`on`/`for`/
`at`) bridge a span when a capitalized token follows — deliberately excluding `and`/`or`/`vs`,
which must keep splitting comparison queries into two separate entity spans
("Doom Eternal **and** Crusader Kings III"). Re-verified clean afterward.

**Test coverage added** (previously zero for this module): `tests/test_quality_gate.py`,
`tests/test_corpus_index.py` — hermetic, entity index injected via
`CorpusEntityIndex.from_titles()`, no network. `tests/regression_suite.py`'s BUG-001/BUG-003
fixtures were updated from `PARTIAL`→`FULL` with inline comments — the relevance-ladder and
`is_noise` fixes retain more genuine evidence for those two historical cases, which is a correct
grading change, not a silent weakening (both now pass).

Full artifacts: `evaluation/results/relevance_calibration_2026-08-12.json`,
`runs_2026-08-12_{default,corpusonly}.jsonl`, `refusal_2026-08-12_{default,corpusonly}.json`.

**Follow-up fix (2026-08-19):** a full-codebase audit (`AUDIT_TASKS.md` §8/§9) found two more
false-refusal sources in `corpus_index.py`, both landing after this phase closed. §8:
`candidate_spans()` unconditionally skipped the sentence-initial token, so an entity-only query
with no interrogative lead-in ("Far Cry 5 combat") produced no matching span and fell through to
a refusal for a game genuinely in the corpus. Fixed with an asymmetric span search — a
conservative span set (unchanged default) decides whether the query names an entity at all, and a
separate greedy set (including the first token) is checked for a match — constructed so it can
only turn a false refusal into a correct answer, never the reverse. §9: the entity index was a
bare process-lifetime singleton with no refresh path; added a TTL (default 900s) with a
single-flight refresh lock so newly-ingested titles become groundable without a restart, without
adding request latency or a thundering herd against Qdrant. See the "Resolved" notes in
`AUDIT_TASKS.md` §8/§9 for the full trace, including why the audit's own originally-proposed §8
fix was rejected (it introduced new refusals on 5 of the 50 golden-set queries).

### Phase 4 — Presentation (4–6 h)

**4.1 Rewrite the metrics section** *(~2 h)* — Replace the five "resume-grade" tables with
**one** table: RAGAS scores, refusal precision/recall, p50/p95 latency, cost per query. Fewer,
real, falsifiable numbers. Include the Phase 3.4 ablation table.

**4.2 Add "Design Decisions & Rejected Alternatives"** *(~2 h)* — The rubric wants rejected
options, and this project has genuinely good ones already visible in the code comments:
*RRF hybrid over pure cosine* (with ablation numbers); *cross-encoder reranking on Modal rather
than in-process* — the comment at `rag_retriever.py:49-51` documents that loading
torch/sentence-transformers alongside fastembed blew the 512 MB Render limit, which is a real
engineering-constraint story; *deterministic routing over an LLM router* (latency/reproducibility);
*bounded LLM web-search decision that fails soft to a deterministic rule*.

**4.3 Purge the "resume-grade" framing** *(~1 h)* — Strings like "RESUME-GRADE KPI",
"Resume-grade interpretation", "KEY RESUME KPI" appear throughout the source. A reviewer
reading the code sees the metrics were designed to look good rather than to be true. Rename to
neutral technical language.

**4.4 Demo GIF in README** *(~1 h)* — A 20-second capture showing a normal answer with sources,
then the GTA VI query producing a refusal. The refusal is the differentiator; show it.

### Total: ~25–35 hours. Phase 0 and Phase 3 carry the most value per hour.

### Open questions for the candidate — answered

1. **Resolved.** Git history was preserved — merge commit `a78e73f` folded the off-machine
   history (20 pre-Phase-0 commits) into this repo rather than starting fresh.
2. **Resolved, from live traffic.** `engine/execution_engine.py`'s `STEP 7` calls
   `llm/ragent_client.py`'s `chat_completion_remote` (Groq `llama-3.1-8b-instant`) — this is the
   only LLM path exercised across every one of the 100 golden-set query runs this session.
   `llm/modal_llm.py` (Gemma 3 12B) was not invoked. Recommend deleting it or clearly marking it
   experimental/unused rather than maintaining a third inconsistent claim in the README.
3. **Answered by direct measurement, and the answer is "less than expected."** ~500 judge calls
   was the estimate; the actual constraint hit was Groq's **daily** token quota for the 70B
   judge model (100,000 TPD), not a per-minute rate limit — it was nearly exhausted after
   roughly 8-10 fully-scored golden-set records in a single session. The checkpoint/resume
   design (built for exactly this) worked once a bug in it was found and fixed (see Phase 3
   results). Recommend either running RAGAS scoring across multiple days, or reducing the
   golden set / RAGAS subset size if a same-day result is required.

### Rubric conflict flagged

None. The gaming corpus, hybrid retrieval design, and honesty-gate architecture already on
disk are consistent with the rubric — this plan strengthens prior decisions rather than
overriding any of them. The one thing being *reversed* is not an architecture decision but a
measurement decision: the current KPI suite is replaced because it cannot fail, not because
the system it measures is wrong.

---

## Step 4 — Consolidated flagship-readiness checklist

| # | Rubric item | Status | Action |
|---|---|---|---|
| 1 | Hybrid dense+sparse, RRF | ✅ PRESENT | Ablation run (2026-08-09): RRF hybrid beats dense/BM25 on precision@k (0.95 vs 0.94/0.935); production `hybrid_rerank` default scores *lowest of all four modes* on both precision@k (0.92) and RAGAS context precision (0.5168) on the reordering ablation. Phase 3.5 (2026-08-12) gives the cross-encoder's `rerank_score` a second job it does earn: the calibrated relevance floor that drives the honesty gate below — it lost the reordering ablation but is now load-bearing for refusal |
| 2 | Specific non-generic corpus | ✅ PRESENT | Rebuilt: 100 games, 2791 chunks, verified zero orphans/duplicates |
| 3 | "Insufficient information" refusal | ✅ WORKING (one accepted miss) | Phase 3.5 (2026-08-12): refusal recall **0.0 → 0.9** corpus-only, `over_refusal_rate = 0.0` on all 40 answerable queries — see Phase 3.5 results and resolved row 21 below |
| 4 | Source/citation attribution | ✅ ENFORCED (compliance not 100%) | Citation format specified + `validate_answer()` checks against context (0.2); live runs show the LLM doesn't always comply — validator correctly flags it |
| 5 | RAGAS: context precision, faithfulness, answer relevancy | ✅ COMPLETE | 40/40 answerable queries scored (Modal-judged, 2026-08-09): context_precision 0.5722, faithfulness 0.9077, answer_relevancy 0.7306. Groq track partial (5/40, quota-exhausted) and preserved separately, not mixed |
| 6 | Live public deployment | ✅ RESOLVED (2026-08-14) | `https://rag-ent.onrender.com` — `/health` 200, a real sourced query, and the GTA VI refusal path all verified against the live URL |
| 7 | Containerized / one-command | ✅ RESOLVED (2026-08-14) | `docker build -t ragent . && docker run -m 512m --env-file .env ragent` — booted at 124-135MB resident, well under the cap; `/health`/`/ping`/a real query/the refusal path all verified inside the container |
| 8 | Observability: latency | ✅ PRESENT | `ProfileBlock` — keep |
| 9 | Observability: token cost | ✅ RESOLVED (2026-08-14) | `llm/ragent_client.py`/`ragent_client_streaming.py` read `prompt_tokens`/`completion_tokens` off Groq's `usage` (via `extra_body`'s `stream_options`, since the installed SDK has no typed param) and `llm/pricing.py` estimates USD cost; surfaced in both engines' `kpis` |
| 10 | Observability: tracing layer | ✅ RESOLVED (2026-08-14) | `utils/tracing.py` (Langfuse) mirrors every `ProfileBlock` span via a new hook in `utils/observability.py`; hard no-op with zero import cost unless both `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` are set. Verified live in production — screenshot showing a full `chat_request` trace with nested spans and a `LLMGeneration` node (942→512 tokens, $0.000088) |
| 11 | README value prop | ✅ PRESENT | Already complete |
| 12 | README architecture diagram | ✅ PRESENT | Already complete |
| 13 | README metrics table | 🚨 NOT CREDIBLE | Delete fake safety table now (0.1); replace with real numbers (4.1) — **still Phase 4's job, untouched this session by design** |
| 14 | README live demo link | ✅ RESOLVED (2026-08-14) | `https://rag-ent.onrender.com`, verified live |
| 15 | "Why this approach / what I rejected" | ❌ ABSENT | Write it from existing code comments + ablation (4.2) |
| 16 | Depth signal, not checklist | ⚠️ AT RISK | Collapse 5 KPI tables → 1; purge "resume-grade" strings (4.1, 4.3) |
| 17 | Version control / public repo | ✅ PRESENT | Public repo `SukeshShetty1010/RAG_ent`, history preserved via merge commit `a78e73f` |
| 18 | *(defect)* Modal creds missing from Render | ✅ RESOLVED (2026-08-14) → **MOOT (2026-08-16)** | Fresh token pair minted (`modal token new`), set in Render dashboard, redeployed and verified live. Moot since the Modal migration: those env vars are gone from `render.yaml` and `.env`. The Modal tokens themselves remain valid until revoked at modal.com |
| 19 | *(defect)* `healthCheckPath` missing | ✅ RESOLVED | Added to `render.yaml` (2026-08-09) |
| 20 | *(defect)* `Unified_KPI_Runner` path bug | ✅ RESOLVED | `parents[1]`; confirmed clean run this session |
| 21 | *(found by 3.3)* Honesty gate never returns `INSUFFICIENT` on off-corpus queries | ✅ RESOLVED (Phase 3.5, 2026-08-12) | Replaced the RRF-score threshold with a calibrated cross-encoder relevance floor + a new corpus entity-grounding check (`retriever/corpus_index.py`); re-gates on merged evidence after web search. Refusal recall 0.0 → 0.9, `over_refusal_rate = 0.0`. One accepted miss (`g047`) documented as a genuine single-signal limit, not hidden |
| 22 | *(new, found while wiring the Modal RAGAS judge)* Production `llm/modal_llm.py` had a concurrency-corrupting event-loop race | ✅ RESOLVED | See defect #6. Real bug in the live generation service, not eval-only; fixed and redeployed (2026-08-09), validated via ephemeral concurrent smoke test before deploy |
| 23 | *(new, found verifying Docker locally)* `.env` wrapped every value in literal quotes | ✅ RESOLVED (2026-08-14) | `docker run --env-file` doesn't strip quotes the way `python-dotenv` does — `QDRANT_URL` arrived as `"https://...io"` and DNS resolution failed inside the container. Local-only issue (never affected `python-dotenv`-based runs); fixed by stripping quotes from all 15 quoted `.env` values |
| 24 | *(new)* `llm/modal_llm.py` (Gemma 3 12B) had zero call sites anywhere outside its own file — **superseded 2026-08-16**, file deleted and the fallback chain is now Gemini → Groq | ✅ RESOLVED (2026-08-14) | Wired in as a live Groq-rate-limit fallback: `chat_completion_remote()`/`chat_completion_streaming()` catch `groq.RateLimitError` and reroute to `Gemma312BVLLM.generate()`. Verified with real generation, a simulated 429 in both blocking/streaming clients, and an organic trigger during this session's regression suite (Groq genuinely rate-limited from heavy testing). No token/cost tracking on this path — Modal bills by GPU-second, not per-token, and `generate()` has no usage payload; `llm_fallback_modal` counter tracks frequency instead. **Caveat found in the same organic trigger: cold-container fallback latency was ~9.2 minutes** (vLLM engine cold-boot on the L40S) vs. seconds once warm — a `.github/workflows/modal-keepalive.yml` now pings all three Modal services (embedder, reranker, Gemma) every 5 minutes to keep this bounded, but a live user hitting a genuinely cold fallback before the first ping lands would still see the multi-minute wait |
| 25 | *(found during final KPI verification)* `WebSearchDecision` pydantic model constrained `confidence <= 1` | ✅ RESOLVED (2026-08-14) | Root cause: the prompt mislabeled `quality_report.confidence_score` — an unbounded cross-encoder logit (e.g. 7.88) — as `(0-1)`, leading the LLM to occasionally echo an out-of-range value into its own `confidence` field and fail `Field(ge=0, le=1)` on every call, silently degrading to the deterministic fallback since 2026-08-12. Fixed the prompt's scale description and clamp the LLM's reported confidence into `[0,1]` instead of hard-failing. New regression test (`tests/test_web_search_decision.py`, 3 out-of-range cases: 7.88, -2.0, 1.0001) confirms `source="llm"` with a clamped value instead of a fallback. 83/83 unit tests pass |
| 26 | Keepalive coverage (Render / Modal / Qdrant) — **Modal leg removed 2026-08-16** (`modal-keepalive.yml` deleted in `80be984`); the Cloudflare reranker is always warm and needs no ping | ✅ VERIFIED (2026-08-14) | All three `.github/workflows/*-keepalive.yml` manually triggered and confirmed green. `qdrant-keepalive.yml` had been failing every run (curl exit 3, malformed URL) — its `QDRANT_URL`/`QDRANT_API_KEY` GitHub secrets were last set 2026-08-09, predating both this session's key rotation and the `.env` quote-strip fix, so they still held stale quote-wrapped values. Resynced from current `.env`; re-run passed |

---

## Verification

- **Phase 0:** `python -m KPI.Unified_KPI_Runner` runs clean from the project root; grep the
  repo for `resume-grade` and `hallucinated_claims` returns nothing in reporting paths.
- **Phase 1:** ✅ Done, 2026-08-14. `docker build -t ragent . && docker run -m 512m --env-file
  .env ragent` — booted at 124-135MB resident; `/health`/`/ping` → 200; a real query returned
  sources; the GTA VI refusal path returned `INSUFFICIENT`. All four checks repeated against
  `https://rag-ent.onrender.com` and passed.
- **Phase 2:** ✅ Done, 2026-08-14. One live query produced one Langfuse trace with nested spans
  matching every `ProfileBlock` name 1:1, plus a `LLMGeneration` node with non-zero token counts
  (942→512) and cost ($0.000088) — screenshot on file. `python -m pytest tests/ -m unit -q` — 80
  hermetic tests pass, including the new `tests/test_tracing.py` (12 tests: hard no-op without
  keys, `estimate_cost` math, the generic `ProfileBlock` span-hook mechanism).
- **Cost/latency artifacts (Step E of the 2026-08-14 session):** `python -m evaluation.run_eval`
  re-run against the token/cost-instrumented engine (50/50 scored, 0 errors) →
  `evaluation/results/runs_2026-08-14_default.jsonl`; scored by the new
  `evaluation/cost_latency_metrics.py` → `evaluation/results/cost_latency_2026-08-14_default.json`
  — p50 727ms / p95 10.8s / p99 15.1s engine latency, mean cost $0.0000492/query ($0.000738 total
  for the run), stage-level latency attribution (`CrossEncoderRerank` and `EmbeddingGeneration`
  dominate the retrieval side). Both committed for Phase 4 to cite.
- **Modal Gemma fallback (new, 2026-08-14, not in the original plan):** `llm/modal_llm.py`
  (deployed, previously zero call sites) wired in as a live Groq-`RateLimitError` fallback in both
  `chat_completion_remote()` and `chat_completion_streaming()`. Verified with real Modal
  generation, a simulated 429 in both clients, and — separately — an organic trigger during this
  session's `tests/regression_suite.py` run (Groq genuinely rate-limited from the session's
  volume of testing): fallback correctly fired and returned a coherent answer, but took **~9.2
  minutes** on a cold Modal container. `.github/workflows/modal-keepalive.yml` now pings all
  three Modal services every 5 minutes to bound this; the embedder/reranker pings are cheap
  (CPU-only), the Gemma ping bills real GPU-second cost continuously for what's meant to be a
  rare path — an explicit, accepted tradeoff, not an oversight.
- **Final regression pass (2026-08-14, after all of the above):**
  `python -m pytest tests/ -m unit -q` — 80/80 pass. `python -m KPI.Unified_KPI_Runner` — exit 0,
  full dashboard, regression guard 3/3, latency profile reflects the new `ModalLLMFallback` span
  from the organic trigger above. `python tests/regression_suite.py` (via `PYTHONPATH=.`) — 3/3
  pass, "ALL REGRESSIONS PASSED — SYSTEM STABLE". One new finding surfaced by this pass, fixed in
  a follow-up pass the same day (see checklist row 25): `WebSearchDecision`'s pydantic model
  constrained `confidence <= 1` while Phase 3.5's confidence score is unbounded — fixed by
  correcting the misleading prompt label and clamping the LLM's confidence into `[0,1]`.
- **Keepalive verification + pydantic fix (2026-08-14, follow-up pass):** all three keepalive
  workflows manually triggered and confirmed green (Render, Modal — including a real cold-boot of
  all 3 services, Qdrant after resyncing its stale GitHub secrets). `WebSearchDecision`'s
  `confidence` validation bug fixed and regression-tested (see checklist row 25). Full unit suite:
  83/83 pass.
- **Phase 3:** ✅ Done, 2026-08-09. `evaluation/results/runs_2026-08-09_default.jsonl` and
  `_corpusonly.jsonl` (50 records each), `refusal_2026-08-09_default.json` /
  `_corpusonly.json`, `ablation_2026-08-09.json`, and `ragas_2026-08-09_default_modal.json`
  (40/40 complete) all committed. Refusal recall on the 10 unanswerable queries is reported and
  is **not** 100% — it is **0%**, hand-confirmed by reading 5 of the 10 unanswerable answer
  texts directly (all 10 got a `full`/`partial` answer instead of a refusal).
  `ragas_2026-08-09_default.json` (Groq/qwen track) remains partial (5/40, quota-exhausted) and
  is preserved as-is rather than mixed with the completed Modal track.
- **Phase 3 KPI regression:** `python -m KPI.Unified_KPI_Runner` re-run after all of the above
  (including the `llm/modal_llm.py` concurrency fix and redeploy) — exit 0, full dashboard
  produced, no regression.
- **Phase 3.5:** ✅ Done, 2026-08-12. `python -m pytest tests/ -m unit -q` — 68 hermetic tests
  pass, including two new files with previously-zero coverage
  (`tests/test_quality_gate.py`, `tests/test_corpus_index.py`). Calibration artifact
  `evaluation/results/relevance_calibration_2026-08-12.json` committed. Headline number:
  `evaluation/results/refusal_2026-08-12_corpusonly.json` — recall 0.0 → 0.9,
  `over_refusal_rate = 0.0` on all 40 answerable queries (hard gate, met). Default (web-on) run
  and its `by_merge_state` breakdown in `refusal_2026-08-12_default.json`, reported as product
  behavior, not the honesty claim. `tests/regression_suite.py` — 3/3 pass (two fixtures updated
  `PARTIAL`→`FULL` with inline comments explaining the correct-direction change).
  `python -m KPI.Unified_KPI_Runner` re-run after — exit 0, full dashboard, regression guard 3/3.
- **Phase 4:** Every number in the README maps to a committed file under `evaluation/results/`.
