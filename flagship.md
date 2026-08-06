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
| Reranking | Cross-encoder on Modal reorders fused candidates into `rerank_score`; original RRF `score` deliberately preserved for the quality gate; fail-soft to RRF order on error | `retriever/rag_retriever.py:188-221` |
| Domain corpus | Gaming. Three source APIs (RAWG identity, IGDB relational metadata, GameSpot editorial), 5 Qdrant collections, editorial chunking under a written contract (~500 tok / 50 overlap, no-orphan rule) | `data/`, `ingest/`, `vector/create_schema.py`, `chunking/chunk_contract.md` |
| Corpus volume | 50+ games indexed *[candidate]* | — |
| Refusal path | `CapabilityAssessor.assess()` returns `INSUFFICIENT` on empty evidence or `QUALITY_EMPTY`; engine hard-guards generation behind `if capability != AnswerCapability.INSUFFICIENT:` | `agent/capability/capability_assessor.py:58-59`; `engine/execution_engine.py:185` |
| Citation attribution | Context blocks injected as `[Source: {title} | Type: {type}]`; every task template instructs "Cite sources"; frontend renders a "Sources (Evidence)" panel with `source_title` + snippet | `agent/prompt_templates.py:52-55, 105/121/138/152/167`; `frontend/src/app/page.tsx:257-267` |
| Evaluation | **Homegrown deterministic metrics only.** No RAGAS. Test set = 6 hardcoded queries (`TRAFFIC`) + 2 in an `__main__` block | `tests/evaluation_metrics.py`; `KPI/Faith_Fair_KPI.py:30-38`; `tests/evaluation_runner.py:187-214` |
| Observability | Homegrown `MetricsRegistry` + `ProfileBlock` (thread-safe, nested wall-clock). **In-process only** — no export, no persistence, no token/cost capture | `utils/observability.py`; no `token`/`cost`/`usage` match in `engine/execution_engine.py` |
| Containerization | Multi-stage Dockerfile (Node 20 static export → python:3.11-slim), `HEALTHCHECK`, `PORT`-aware CMD | `Dockerfile` |
| Deployment | `render.yaml` exists. **Not deployed** *[candidate]*. No `.git` directory in this folder; repo exists on another machine *[candidate]* | `Test-Path .git` → `False`; `render.yaml` |
| README | Value prop, two Mermaid diagrams, five metrics tables, quick-start, deployment notes. **No live demo link** (`YOUR_USERNAME` placeholder at line 278). **No "what I rejected" section** | `README.md` |

### Defects found during inspection (not rubric items, but blocking)

These matter because the README publishes these numbers as headline achievements.

1. **The safety KPIs cannot fail.** `hallucinated_claims` is initialised to `0` and **never
   incremented** (`KPI/Faith_Fair_KPI.py:69`); "Unsafe Outputs Produced | 0" is a **hardcoded
   string literal** (`:177-179`); all three capability branches print `Safe` (`:105-121`); and
   `honest_rate = (full + partial) / total` is tautological — every non-refusal counts as
   honest by definition. The GTA VI safety-trap query passes regardless of what it outputs.
   The README's "Honest Answer Rate 100.00% / Hallucinated Claims 0" is therefore an artifact
   of the scoring code, not a measurement.
2. **Grounding fidelity is dead code with a format mismatch.** `calculate_grounding_fidelity`
   matches the regex `\(Source:\s*'([^']+)'\)` (`tests/evaluation_metrics.py:175`), but the
   engine's prompt templates only say "Cite sources" with **no format specified** — the strict
   `(Source: 'X')` format exists only in the unused CLI helper `format_llama3_prompt`
   (`retriever/rag_retriever.py:263-264`). So fidelity scores ~0, and it is computed at
   `Faith_Fair_KPI.py:97` then **thrown away** at `:126-128` (the branch body is `pass`).
3. **Deployment will crash on boot as configured.** `retriever/rag_retriever.py:46-53` calls
   `modal.Cls.from_name(...)` at **import time**, and `api/main.py:33` instantiates the engine
   at module import. But `render.yaml` declares no `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, and
   the Dockerfile has no Modal credential handling. First request → import → unauthenticated
   Modal lookup → boot failure.
4. `render.yaml` has **no `healthCheckPath`**, despite the README claiming it defines one.
5. `KPI/Unified_KPI_Runner.py:16` uses `parents[2]`, correct only from the file's old
   `tests/KPI/` location; it now resolves *outside* the project root.

---

## Step 2 — Gap analysis

### Technical substance

| Rubric item | Status | Basis |
|---|---|---|
| Hybrid dense + sparse, RRF-fused | **PRESENT (exceeds)** | `rag_retriever.py:121-147` + rerank stage `:188-221` |
| Specific non-generic corpus | **PRESENT** | 3-API gaming corpus, 50+ games, 5 collections |
| Explicit "insufficient information" handling | **PRESENT** | `capability_assessor.py:58-59` + hard guard `execution_engine.py:185` |
| Source/citation attribution in outputs | **PRESENT (weak enforcement)** | Sources injected + rendered in UI; but no enforced citation format, and nothing validates that emitted citations match retrieved titles |
| Quantified RAGAS-equivalent eval (context precision, faithfulness, answer relevancy) | **ABSENT** | No `ragas` dependency; homegrown metrics are capability-label bookkeeping, not faithfulness; test set is 8 queries total |

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

### Phase 0 — Stop the bleeding (2–3 h)

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

### Phase 1 — Get it live (3–5 h)

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

### Phase 2 — Real observability (4–6 h)

**2.1 Langfuse** *(~4 h)* — Free-tier Langfuse Cloud; `pip install langfuse` (thin SDK, safe
for the 512 MB cap — **do not** add `torch`/`sentence-transformers`, per the deployment
constraint). Wrap the 7 engine steps as spans on a single trace, mirroring the existing
`ProfileBlock` names so both layers agree. Capture per-trace: task type, capability verdict,
quality-gate status, chunk count, whether web fallback fired, and — new — **token counts and
cost from the Groq response's `usage` field**, which is currently discarded.

**2.2 Keep `MetricsRegistry`** — Do not rip it out. It feeds the KPI runners and is a genuine
"I built this myself" talking point. Langfuse is the external trace layer; they coexist.

### Phase 3 — Real evaluation (8–12 h) ← *the highest-value work in this plan*

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

### Open questions for the candidate

1. **Does the off-machine repo have history worth preserving,** or is a fresh `git init` +
   single initial commit acceptable? This changes Phase 1.1 from 30 min to potentially 2 h.
2. **Is `google/gemma-3-12b-it` on Modal still a live path, or is Groq now the only LLM?**
   `structure.md:36` says Qwen 2.5 7B, the README says Gemma 3 12B — these already disagree.
   If Modal LLM is dead, delete it rather than maintaining a third inconsistent claim.
3. **Do you have Groq/Modal budget for ~500 judge calls** (50 queries × 3 RAGAS metrics ×
   retries)? If not, cut the golden set to 30 queries; the plan still holds.

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
| 1 | Hybrid dense+sparse, RRF | ✅ PRESENT | Already complete — add ablation proof (3.4) |
| 2 | Specific non-generic corpus | ✅ PRESENT | Already complete |
| 3 | "Insufficient information" refusal | ✅ PRESENT | Already complete — add falsifiable metric (3.3) |
| 4 | Source/citation attribution | ⚠️ WEAK | Enforce citation format + validate against context (0.2) |
| 5 | RAGAS: context precision, faithfulness, answer relevancy | ❌ ABSENT | Golden set + RAGAS, committed results (3.1, 3.2) |
| 6 | Live public deployment | ❌ ABSENT | Fix Modal lazy-binding, deploy to Render, cron keep-alive (1.2–1.4) |
| 7 | Containerized / one-command | ⚠️ UNVERIFIED | Actually build and boot the image locally (1.3) |
| 8 | Observability: latency | ✅ PRESENT | `ProfileBlock` — keep |
| 9 | Observability: token cost | ❌ ABSENT | Capture Groq `usage`, send to Langfuse (2.1) |
| 10 | Observability: tracing layer | ❌ ABSENT | Langfuse spans over the 7 engine steps (2.1) |
| 11 | README value prop | ✅ PRESENT | Already complete |
| 12 | README architecture diagram | ✅ PRESENT | Already complete |
| 13 | README metrics table | 🚨 NOT CREDIBLE | Delete fake safety table now (0.1); replace with real numbers (4.1) |
| 14 | README live demo link | ❌ ABSENT | Replace `YOUR_USERNAME` placeholder after 1.1 |
| 15 | "Why this approach / what I rejected" | ❌ ABSENT | Write it from existing code comments + ablation (4.2) |
| 16 | Depth signal, not checklist | ⚠️ AT RISK | Collapse 5 KPI tables → 1; purge "resume-grade" strings (4.1, 4.3) |
| 17 | Version control / public repo | ❌ ABSENT here | Install git, `git init`, reconcile, push (1.1) |
| 18 | *(defect)* Modal creds missing from Render | 🚨 BOOT BLOCKER | Add tokens to `render.yaml` + lazy binding (1.2) |
| 19 | *(defect)* `healthCheckPath` missing | ❌ ABSENT | Add to `render.yaml` (1.2) |
| 20 | *(defect)* `Unified_KPI_Runner` path bug | 🐛 BROKEN | `parents[2]` → `parents[1]` (0.3) |

---

## Verification

- **Phase 0:** `python -m KPI.Unified_KPI_Runner` runs clean from the project root; grep the
  repo for `resume-grade` and `hallucinated_claims` returns nothing in reporting paths.
- **Phase 1:** `docker build -t ragent . && docker run -p 10000:10000 --env-file .env ragent`;
  `curl localhost:10000/health` → 200; a real query returns sources in the UI. Then the same
  two checks against the public Render URL.
- **Phase 2:** One query produces one Langfuse trace with 7 nested spans and non-zero token
  counts.
- **Phase 3:** `evaluation/results/ragas_<date>.json` exists with all three metrics over 50
  queries; refusal recall on the 10 unanswerable queries is reported and is **not** 100% by
  construction — confirm by hand-checking at least 3 of those traces.
- **Phase 4:** Every number in the README maps to a committed file under `evaluation/results/`.
