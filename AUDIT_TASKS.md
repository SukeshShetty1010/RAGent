# RAGent — Full-Codebase Audit & Task List

**Audit date:** 2026-08-18
**Scope:** every module on the request path (`api/` → `engine/` → `agent/` → `retriever/` → `llm/` → `utils/` → `frontend/`), plus ingest, evaluation, config, Docker, and CI.
**Method:** read end to end, line by line. Claims about live behaviour were measured against the real Qdrant corpus and the real `.env`, not inferred.
**Test suite at time of audit:** `159 passed, 3 skipped` (`python -m pytest tests/`).

The system runs and answers queries. Nothing here is a crash. Everything here is a case where the code **executes successfully while not delivering the behaviour it was written to deliver**, or a case where a component was built and then never connected to anything.

---

## Status update — 2026-08-19

**T2 and T20 fixed** together, commit [`b70ccfa`](https://github.com/SukeshShetty1010/RAG_ent/commit/b70ccfa) on `main`. Test suite now `168 passed, 3 skipped` (up from 159 — 9 net new tests; two pre-existing `live`-marked tests in `test_qdrant_rebuild.py` fail locally for unrelated reasons, no `.env`/Qdrant credentials in this environment).

These two were fixed together, not independently, because they're causally coupled: §2's own fix note below already says a `rerank_score` sort produces a list "sorted incoherently across two different scales" unless §20 is fixed first — §20 is exactly how that half-scored list gets produced. See the "Resolved" notes inside §2 and §20 for what actually shipped, which differs from each section's originally-proposed snippet in two deliberate ways:

- §2 shipped an all-or-nothing `is_fully_reranked()` / `relevance_key()` pair instead of the per-chunk `_relevance()` fallback the fix snippet proposed — the snippet's own approach mixes scales within one sort, which is precisely the hazard the surrounding prose warns against.
- §20's guard was placed in `_rerank_scores()` (the single dispatch point shared by `_rerank` and `score_relevance`), not in `_rerank()` as proposed, because `score_relevance()` has the identical exposure via `orchestrator.py:229`. The provider table in §20 also had two errors, corrected there: `hfspace` already guards against a short result, and `local` — not `hfspace` — is the actually-unguarded default provider.

**T8 and T9 fixed** together (same day, follow-up pass). Both live in `retriever/corpus_index.py` and both feed the same consumer, `RetrievalQualityGate.evaluate()` — currently the only live refusal path in production per §1, so their false positives were the entire refusal surface. See the "Resolved" notes inside §8 and §9. §8 in particular shipped a materially different fix than the snippet originally proposed in that section — the proposed one-liner was hand-traced and found to introduce its own new false refusals (`"What's..."`, `"Tell me..."`, `"Compare X and Y"` queries), so a monotone verdict/match span-set split was used instead. This fix should land *before* §1 (relevance floor calibration) and §17 (evaluation re-runs), since `evaluation/calibrate_relevance.py` uses `assess_grounding()` to partition golden-set queries and this change alters some of those verdicts.

**T5 and T6 fixed** together (same day, follow-up pass). Causally coupled — §6's own text notes that under §5, a concurrent request's Gemini counter can flip the (already wrong) model attribution, so §6 could not be fixed durably without §5 first. `MetricsRegistry` (`utils/observability.py`) is now scoped per-thread via `threading.local()`, mirroring `utils/tracing.py`'s existing pattern; the answer-serving model is now recorded explicitly (`answer_model` categorical) at the point each provider actually produces the answer, rather than inferred from provider-usage counters, and is now surfaced through `kpis["answer_model"]` and a UI tile, not just Langfuse. See the "Resolved" notes inside §5 and §6. Test suite now `190 passed, 3 skipped` (up from 179 — 11 net new tests; the two pre-existing `live`-marked `test_qdrant_rebuild.py` failures are unrelated, no `.env`/Qdrant credentials in this environment).

**T10 and T11 fixed** together (same day, follow-up pass), commit history on `main`. Causally coupled — both live in `agent/task_router.py` / `retriever/strategy_selector.py` and both concern the same underlying question: who owns the "can this query reach the web?" decision, and by what rule. `RouterDecision`'s three dead fields (`retrieval_strategy`, `web_search_allowed`, `max_results`) were deleted rather than consumed — a repo-wide grep confirmed nothing read them outside `strategy_selector.py`'s own unused test-harness fixtures, and the two independent derivations already disagreed. Separately, `StrategySelector.select()` now sets `allow_web_fallback=True` for COMPARISON, LISTICLE, and both FACTUAL branches whenever `IntentSignal.TEMPORAL` is present in `decision.intent_signals` (previously hardcoded `False`, reachable only via `TaskType.OPEN`), so a query like *"latest patch notes for Assassin's Creed Valhalla"* (`FACTUAL` + `TEMPORAL`) can now reach `orchestrator.py`'s temporal web-fallback gate. `quality_report.has_temporal_signal` (chunk-content-level, `retriever/quality_gate.py`) is a distinct signal from `IntentSignal.TEMPORAL` (query-level) and was intentionally left untouched — T11's fix only removes the task-type gate that blocked the temporal check from ever being evaluated for non-OPEN tasks. See the "Resolved" notes inside §10 and §11. Test suite now `201 passed, 3 skipped` (up from 190 — 11 net new tests in `tests/test_strategy_selector.py`, a file that did not exist before; the two pre-existing `live`-marked `test_qdrant_rebuild.py` failures are unrelated, no `.env`/Qdrant credentials in this environment).

Remaining 15 tasks are unchanged from the original audit below.

---

## Task checklist

Ordered by impact. Items 1–3 are the ones that change what the user actually receives.

- [ ] **T1** — Calibrate the Cloudflare reranker floors; the honesty gate is currently switched off (§1)
- [x] **T2** — Order assembled context by `rerank_score`, not the RRF `score` (§2) — Fixed 2026-08-19
- [ ] **T3** — Finish the last 91 chunks of the Gemini embedding migration (§3)
- [ ] **T4** — Remove the `max_tokens=150` override in `decide_web_search` (§4)
- [x] **T5** — Scope `MetricsRegistry` per request, or accept cross-request KPI contamination (§5) — Fixed 2026-08-19
- [x] **T6** — Fix `last_used_model()` to report the model that served the *answer* (§6) — Fixed 2026-08-19
- [ ] **T7** — Reconcile the two engines: `llm_latency_ms`, trace attributes, refusal string (§7)
- [x] **T8** — Make `candidate_spans()` handle non-interrogative queries (§8) — Fixed 2026-08-19
- [x] **T9** — Allow the entity index to refresh without a process restart (§9) — Fixed 2026-08-19
- [x] **T10** — Either consume `RouterDecision`'s three dead fields or delete them (§10) — Fixed 2026-08-19
- [x] **T11** — Decide whether web fallback should be reachable outside `TaskType.OPEN` (§11) — Fixed 2026-08-19
- [ ] **T12** — Either send `insufficient_prompt()` to the LLM or delete it (§12)
- [ ] **T13** — Render the `stage` SSE events in the UI (§13)
- [ ] **T14** — Decide whether the product is multi-turn; wire history if so (§14)
- [ ] **T15** — Delete `format_llama3_prompt()` (§15)
- [ ] **T16** — Pin `fastembed` in the root `requirements.txt` (§16)
- [ ] **T17** — Re-run the evaluation suite; every stored result predates the current system (§17)
- [ ] **T18** — Narrow `NOISE_KEYWORDS` so ordinary review prose isn't discarded (§18)
- [ ] **T19** — Cancel the engine thread when the SSE client disconnects (§19)
- [x] **T20** — Harden `_rerank()` against a short score list (§20) — Fixed 2026-08-19
- [ ] **T21** — Fix stale comments and docs that describe retired infrastructure (§21)
- [ ] **T22** — Drop the unused `transformers` dependency (§22)
- [ ] **T23** — Invert the two "uncalibrated placeholder" tests once T1 lands (§23)

---

# Part A — Executes, but does the wrong thing

These have measurable effects on answers, refusals, or reported metrics.

---

## §1 — The honesty gate's relevance floor is inert in production

**Severity:** Critical — this is the project's flagship feature, and it is off.
**Files:** `retriever/quality_gate.py:145-168`, `:239-262`; `.env:21`

### What the code does

`.env` line 21 sets:

```
RERANKER_PROVIDER=cloudflare
```

`retriever/quality_gate.py`'s provider-scoped floor table reads:

```python
_FLOORS: Dict[str, Optional[Tuple[float, float]]] = {
    "local":      (-3.0, 2.0),
    "hfspace":    (-3.0, 2.0),
    "cloudflare": None,        # ← active provider
    "voyage":     None,
}
```

`_resolve_floors()` (line 168) returns `None`. `evaluate()` then hits this branch:

```python
if not rerank_scores or floors is None:
    ...
    report = QualityReport(
        status=QualityStatus.QUALITY_OK,     # ← unconditional OK
        reason=f"Evidence present (relevance floor skipped — {cause})",
        confidence_score=avg_score,          # ← avg RRF, not max rerank
        ...
    )
    return report
```

### What was intended

The relevance ladder was supposed to grade evidence three ways: below `REFUSE_FLOOR` → `QUALITY_EMPTY` (refuse), below `WEAK_FLOOR` → `QUALITY_WEAK` (answer partially and say what's missing), otherwise → `QUALITY_OK`. That grading is the entire honesty gate.

The `None` was a deliberate, correct safety decision: Cloudflare's `bge-reranker-base` emits a sigmoid-saturated 0..1 score, and reusing the local model's raw-logit floors (`-3.0`, `2.0`) would make `2.0` unreachable — every query WEAK — and `-3.0` untrippable — refusal impossible. `None` was meant as a temporary "not calibrated yet" marker.

### What is actually wrong

The temporary marker became permanent. `evaluation/calibrate_relevance.py` exists, is written for exactly this, documents Cloudflare's scale in its own docstring, and has **never been run against Cloudflare**. So in production right now:

- Every query that retrieves at least one non-noise chunk returns `QUALITY_OK`.
- `QUALITY_WEAK` is unreachable. `CapabilityAssessor.assess()` therefore never downgrades on weak evidence, so `AnswerCapability.PARTIAL` is only ever produced by the comparison-imbalance and temporal-signal rules.
- Because PARTIAL is nearly unreachable, `agent/prompt_templates.py` never appends the `"Unsupported or Missing Parts:"` instruction, so the model is never told to declare gaps — and `agent/output_validator.py`'s check for that section never has anything to validate.
- The weak-evidence web-search path (`orchestrator.py` STEP 3, the `QUALITY_WEAK` branch) never fires, so `decide_web_search()` — the one deliberately agentic decision in the pipeline — is nearly dead code in production.
- The only surviving refusal path is entity grounding via `retriever/corpus_index.py`.
- `confidence_score` falls back to `avg_score`, the **mean RRF fusion score**, instead of `max_relevance`. The UI renders this as `Confidence: {value * 100}%`, so the dashboard's confidence percentage is currently an averaged RRF fusion score dressed as a probability.

The system is answering with FULL confidence on evidence it has never graded.

### Fix

```bash
# with RERANKER_PROVIDER=cloudflare in the environment
python -m evaluation.calibrate_relevance
```

Read the max-F1 split point out of `evaluation/results/relevance_calibration_<date>.json` and set `_FLOORS["cloudflare"] = (refuse, weak)`. Expect the signal in the tails, not a smooth spread — measured Cloudflare scores are ~0.99990 for a match and ~3.7e-05 for a miss, so the two floors will sit much closer together than the local model's do.

**Do this after §3** — calibrating against a corpus that is 3% still in the old embedding space would bake that mismatch into the floors.

---

## §2 — Reranking is undone before the LLM ever sees the context

**Severity:** Critical — the reranker costs money and latency on every query and currently changes almost nothing about what gets read.
**Files:** `retriever/rag_retriever.py:377-407`; `agent/context_algorithms.py:80-160`; `agent/context_assembler.py:49, 83-105`

### What the code does

Retrieval reranks correctly:

```python
# rag_retriever.py:398-403
for c, s in zip(candidates, rerank_scores):
    c["rerank_score"] = float(s)
candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
```

Two steps later, context assembly re-sorts the same list:

```python
# context_algorithms.py:146-160
def order_factual(chunks):
    return sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
```

`c["score"]` is **the original RRF fusion score**, deliberately preserved by `_rerank()` and never overwritten. `order_comparison` (line 101/103/115) and `order_listicle`'s unordered tail (line 141) sort by the same field.

Then the budget is applied:

```python
# context_assembler.py:49
MAX_CONTEXT_CHARS = 4000
```

### What was intended

`_rerank()`'s docstring is explicit: *"Reorder RRF candidates by cross-encoder relevance ... leaves the original `score` (RRF fusion score) untouched for downstream consumers."* Preserving `score` was right — `quality_gate.py` reads it. The mistake is that the downstream consumer which decides **prompt ordering** picked the preserved field instead of the new one.

### What is actually wrong

Measured against the live corpus (300-chunk sample):

```
n=300  min=81  median=1507  mean=1281  max=2073   chars
chunks over the 4000-char cap: 0 / 300
```

So `apply_character_budget` admits roughly **2–3 chunks** out of the 5–10 retrieved, in first-come order. And "first" is decided by RRF rank, not by the cross-encoder.

Net effect: the reranker's *ordering* is discarded. Its only surviving influence is on which candidates survive the `limit` slice at the end of `_rerank()` — and since `fetch_limit = max(limit*4, 20)` fetches 20+ candidates and keeps `limit` (5–10), a chunk the cross-encoder ranked #1 can easily be pushed out of the 2–3 that fit by chunks that merely rank well on RRF.

One secondary note, currently inert but worth knowing: `apply_character_budget` silently drops any chunk longer than the whole cap (`if content_len > char_cap: continue`). At today's chunk sizes (max 2073) nothing triggers it, but a larger `chunk_size` in `chunking/editorial_chunker.py` would start deleting the best evidence with no log line.

### Fix

Sort by `rerank_score` where it exists, falling back to `score` when the reranker was unavailable (which must keep working — `_rerank()` is fail-soft and deliberately leaves the field unset on failure):

```python
def _relevance(c: Dict[str, Any]) -> float:
    s = c.get("rerank_score")
    return float(s) if s is not None else float(c.get("score", 0.0))
```

Careful: a mixed list (some chunks with a rerank score, some without) sorts incoherently across two different scales. Prefer *all* rerank scores or *none* — and see §20, which is exactly how a mixed list gets produced.

Separately, consider whether `MAX_CONTEXT_CHARS = 4000` still makes sense. It is a Modal/Llama-8B-era number. Gemini Flash Lite has a million-token window; the pipeline retrieves and reranks 5–10 chunks and then throws away 60–70% of them for a budget that no longer binds.

### Resolved — 2026-08-19

Fixed alongside §20 (they're causally coupled — see the status update at the top of this file). Shipped `is_fully_reranked(chunks)` + `relevance_key(chunks)` in `agent/context_algorithms.py` instead of the per-chunk `_relevance()` fallback proposed above: the proposed snippet scores each chunk independently, defaulting to `score` per-chunk when `rerank_score` is missing on *that* chunk — which is exactly the "mixed list sorts incoherently across two scales" hazard this section already warns about, just moved from an accidental outcome into the fix itself. `relevance_key()` instead decides once, over the whole list: `rerank_score` for every chunk if every chunk has one, else `score` for the whole list. All 5 sort sites (`order_comparison` ×3, `order_listicle`'s tail, `order_factual`) now go through it.

Also caught in the process: on a `LOCAL_PLUS_WEB` merge, sorting by `score` was already mixing Qdrant's RRF fusion score (~0.016–0.033) with Tavily's 0..1 relevance (`agent/tools/web_search.py:110`) — a second instance of the same cross-scale hazard, independent of the reranker question. `relevance_key()` fixes this too, since local and web chunks share the reranker's scale once both are scored.

`MAX_CONTEXT_CHARS` was deliberately left untouched (ordering-only fix, by scope decision).

`agent/context_assembler.py` gained observability the section didn't ask for but the fix needed to be verifiable: `context_order_key` (records whether `rerank_score` or `score` won, per assembly call) and `context_chunks_dropped_by_budget`, plus a warning when a chunk exceeds `MAX_CONTEXT_CHARS` and is dropped whole (closing this section's "no log line" secondary note).

Tests: `tests/test_context_ordering.py` (new, 8 tests) — zero test coverage existed for `context_algorithms.py` or `context_assembler.py` before this.

---

## §3 — The embedding migration is incomplete, and production is querying the mismatched corpus

**Severity:** High — silent, partial retrieval degradation on every query.
**Files:** `scripts/migrate_embeddings_to_gemini.py`; `.migration_checkpoint_gemini_embed.json`

### State

| | |
|---|---|
| Total `EditorialChunk` points | 2791 |
| Migrated to `gemini-embedding-001` | **2700** |
| Still holding old E5-base-v2 vectors | **91** |

This session advanced it from 1750 → 2700, then hit the Gemini free-tier daily embedding quota: 429 through the full retry ladder in both `_embed_with_retry` (4 attempts) and `embed_texts` (4 attempts), then `resp.raise_for_status()` raised.

### What is wrong

`retriever/rag_retriever.py` embeds every query with `embed_text(query, task_type="RETRIEVAL_QUERY")` — Gemini space. The 91 unmigrated points are still stored in E5 space. Cosine similarity between vectors from two unrelated embedding spaces is noise, so those 91 chunks are effectively invisible to dense retrieval (or, worse, surface at random). BM25 sparse still works for them, and RRF fusion partly masks the problem, which is exactly why this is silent rather than obviously broken.

CLAUDE.md states the rule plainly — *"must complete before deploying code that queries with Gemini embeddings, or query/stored vectors are mismatched"* — and the deploy went out ahead of it.

### Fix

Tomorrow, after the daily quota resets:

```bash
RAG_env\Scripts\python.exe -m scripts.migrate_embeddings_to_gemini --resume
```

91 chunks is 2 scroll batches; it will finish in under a minute if the quota is clear. Then delete the checkpoint file, and **re-run any calibration or evaluation after this**, not before.

---

## §4 — `decide_web_search` overrides the max-tokens fix it depends on

**Severity:** High — silently converts the agentic decision into its deterministic fallback.
**Files:** `agent/decisions/web_search_decision.py:114`; `llm/ragent_client.py:213`

### What the code does

`chat_completion_decision`'s signature carries a deliberately-raised default:

```python
# llm/ragent_client.py:213
def chat_completion_decision(prompt, max_tokens: int = 320, ...)
```

Its docstring explains the number in detail: the Groq fallback is now a *reasoning* model, hidden reasoning tokens bill against `max_tokens`, Groq rejects `reasoning_effort="none"`, a measured decision burned 88 reasoning tokens before emitting 44 characters of JSON, and — critically — **running out returns empty content rather than an error**.

The only caller passes:

```python
# agent/decisions/web_search_decision.py:114
raw = chat_completion_decision(
    prompt,
    max_tokens=150,          # ← shadows the 320 default
    temperature=0.0,
    response_format={"type": "json_object"},
)
```

### What is actually wrong

The fix was applied at the function definition and the call site was never updated, so the fix has no effect. When the Groq fallback serves this call, the model can exhaust 150 tokens on reasoning, return `""`, and `json.loads("")` raises. The `except Exception` catches it and returns `_deterministic_fallback(...)` tagged `source="deterministic_fallback"`.

That degradation is *by design* fail-soft, which is why it has never been noticed: the pipeline keeps working, the web-search decision just quietly stops being agentic. The only visible trace is `web_decision_source` in the metrics registry, which nothing alerts on.

This bites only on the Groq path — Gemini Flash Lite is not a reasoning model — but the Groq path is precisely the degraded path where you least want a second, hidden degradation stacked on top.

### Fix

Delete the `max_tokens=150` argument. Let the 320 default apply.

---

## §5 — `MetricsRegistry` is a process-wide singleton under a per-request threading model

**Severity:** High — corrupts every reported KPI under concurrency.
**Files:** `utils/observability.py:63-80`; `api/main.py:154`; `engine/execution_engine.py:332`; `engine/execution_engine_streaming.py:499`

### What the code does

`api/main.py` runs each chat request on its own thread:

```python
threading.Thread(target=run_engine, daemon=True).start()
```

Both engines begin a run by wiping the shared registry:

```python
@staticmethod
def _reset_metrics(registry: MetricsRegistry) -> None:
    registry._counters.clear()
    registry._distributions.clear()
    registry._categoricals.clear()
```

`MetricsRegistry.get()` returns one process-wide instance.

### What is actually wrong

The registry is *thread-safe* — every mutation takes `_registry_lock` — but it is not *thread-scoped*. Two overlapping requests share one set of counters. Request B's `_reset_metrics()` wipes request A's in-flight metrics mid-run. Then:

- `kpis["prompt_tokens"]` / `["completion_tokens"]` / `["cost_usd"]` are computed as `avg × count` over the distribution, i.e. the sum across *both* requests
- `kpis["finish_reason"]` takes `max(finish_reasons, key=...)` — the modal value across both requests
- `kpis["answer_truncated"]` derives from that, so one truncated answer can mark a complete one as truncated, or vice versa
- `last_used_model()` reads `llm_provider_*` counters that both requests incremented (see §6)
- `evidence`, `final_answer`, and the stage timings are per-request locals and stay correct — so the answer is right while the numbers next to it are wrong

The comment on `_reset_metrics` in CLAUDE.md notes that skipping the reset leaks counters across requests, and that a bug from exactly this was fixed once. The reset fixed the *sequential* case. The *concurrent* case was never addressed.

`utils/tracing.py` already solves this correctly with `threading.local()` (line 28, with a comment explaining precisely this reasoning). The registry never got the same treatment.

### Fix

Two options:

1. **Thread-local registry** — mirror `tracing.py`: make `MetricsRegistry.get()` return a per-thread instance. Cleanest, and `_reset_metrics()` can then go away entirely. Check every consumer first: `evaluation/`, `KPI/`, and the CLI harnesses read the registry from the main thread and expect process-wide accumulation.
2. **Accept it and document it** — legitimate if this stays a single-user demo. But then the KPI panel should not be presented as per-answer truth.

### Resolved — 2026-08-19

Fixed alongside §6 (they're causally coupled — see the status update at the top of this file). Shipped option 1: `MetricsRegistry.get()` (`utils/observability.py`) now returns a per-thread instance via `threading.local()`, mirroring `utils/tracing.py`'s `_local` exactly, including its comment explaining the per-request-thread reasoning. `_reset_metrics()` was not deleted as originally proposed — it survives as a public `MetricsRegistry.reset()` method, called once at the top of each engine `run()`/`run_streaming()`, because a thread-local instance is only fresh if the thread is fresh, and `reset()` keeps correctness independent of whether a future change pools or reuses worker threads.

Every consumer was checked before making the change, not assumed safe: no `ThreadPoolExecutor`/`concurrent.futures` usage exists anywhere on the request path (only `api/main.py`'s per-request `threading.Thread` and an unrelated `asyncio.to_thread` in `evaluation/ragas_embeddings.py` that never touches the registry); `KPI/System_Performance_KPI.py`, `KPI/Context_Engineering_KPI.py`, `tests/KPI_run.py`, and the CLI harnesses (`retriever/rag_retriever.py`, `agent/tools/web_search.py`, `upsert/upsert_all.py`) all construct the engine and read the registry synchronously on the same (main) thread; `evaluation/` never reads the registry at all.

`KPI/System_Performance_KPI.py` and `KPI/Context_Engineering_KPI.py` were updated to call `registry.reset()` instead of clearing the three private dicts by hand.

Tests: `tests/test_observability.py` gained thread-isolation coverage — two threads writing independently don't leak into each other or the main thread, and the exact §5 failure mode (thread A records, thread B resets and records, thread A's report stays untouched) is now a passing regression test that failed against the pre-fix singleton.

**Found but out of scope:** `KPI/Context_Engineering_KPI.py` runs 5 queries in a loop expecting to aggregate metrics across all of them, but `RageEngine.run()` resets the registry at the start of *every* run — so its aggregate KPIs (e.g. "Prompt Budget Compliance Rate") only ever reflect the last query while `total_runs` stays 5. Pre-existing, not introduced by this fix, and not fixed here since it needs a separate decision about per-run vs. aggregate semantics.

---

## §6 — `last_used_model()` reports the wrong model when the decision and the answer used different providers

**Severity:** Medium — mis-attributed model and cost in Langfuse.
**File:** `llm/ragent_client.py:98-110`

### What the code does

```python
def last_used_model() -> str:
    counters = MetricsRegistry.get().generate_report()["counters"]
    if counters.get("llm_provider_gemini", 0) > 0:
        return GEMINI_MODEL
    if counters.get("llm_provider_groq", 0) > 0:
        return _GROQ_MODEL
    return "unknown"
```

### What is actually wrong

It is named "last used" but implemented as "gemini wins if it was used at all in this request." Within a single request there can be two LLM calls:

1. `decide_web_search()` → `chat_completion_decision()` → succeeds on Gemini → `llm_provider_gemini += 1`
2. answer generation → `chat_completion_streaming()` → Gemini fails → falls back to Groq → `llm_provider_groq += 1`

`last_used_model()` now sees a non-zero Gemini counter and returns `GEMINI_MODEL`. The engine passes that straight into `tracing.record_generation(model=...)` alongside Groq's token counts and Groq's cost. Langfuse records a Gemini generation that never happened, and `llm/pricing.py` has already priced the tokens against whichever model `_record_usage` was given — so the trace is internally inconsistent too.

Under §5 this gets worse: a *different* concurrent request's Gemini counter can flip this.

### Fix

Record the provider that served the answer explicitly rather than inferring it from counters — e.g. have the streaming and blocking clients set a categorical `answer_provider` and read that, or have them return the model name alongside the text.

### Resolved — 2026-08-19

Fixed alongside §5 (see the status update at the top of this file, and §5's own "Resolved" note — the thread-local registry migration was a prerequisite so this fix couldn't be undone by a concurrent request's counters). Shipped the recommended approach: `llm/ragent_client.py` and `llm/ragent_client_streaming.py` now record a categorical `answer_model` explicitly at each point a user-facing answer is actually produced (Gemini success, Groq fallback success, and the mid-stream-failure branch where Gemini tokens already reached the user before it died) via a new `_record_answer_model()` helper, instead of inferring the model from the `llm_provider_*` counters. `chat_completion_decision()` now records a separate `decision_model` categorical so a Gemini-served web-search decision can no longer masquerade as the model that wrote the answer.

`last_used_model()` was replaced with `answer_model()` (`MetricsRegistry.get().last_label("answer_model") or "unknown"`) rather than kept as a compatibility alias — "last used" was the name that licensed the original bug, and there were only 4 call sites plus one `README.md` mention to update, no external consumer. `MetricsRegistry` gained `last_label(name)`, the categorical mirror of the existing `last(name)` for distributions.

Surfaced past Langfuse: both engines now capture `answer_model_used` and add `kpis["answer_model"]`, and the frontend (`frontend/src/app/page.tsx`) renders it as a new "Answer Model" tile in the KPI panel (`Metric` gained an optional `className` prop to let the tile span two columns, since model names like `gemini-flash-lite-latest` don't fit the default tile width).

Tests: `tests/test_answer_model_attribution.py` (new) — covers the exact §6 scenario end to end (decision served by Gemini, answer falls back to Groq → `answer_model()` must return the Groq model, which failed against the pre-fix code), plus the Gemini-served, decision-only, and mid-stream-failure cases. `last_label()` gained unit coverage in `tests/test_observability.py`.

---

## §7 — The two engines have drifted apart

**Severity:** Medium — the blocking engine and the SSE engine no longer produce the same contract.
**Files:** `engine/execution_engine.py:173, 250, 254`; `engine/execution_engine_streaming.py:256, 343, 389`

Three concrete divergences:

### 7a. `llm_latency_ms` is reported on a failed generation

The streaming engine assigns it unconditionally after the try/except:

```python
# execution_engine_streaming.py:343
llm_latency_ms = (time.perf_counter() - step_start) * 1000.0
```

If generation raised, `llm_ran` is `False` but `kpis["llm_latency_ms"]` is a real number. The blocking engine assigns it only inside the success path, so on failure it stays `None`. Same failure, two different KPI payloads. The UI's `ms()` helper renders the streaming one as a plausible duration for an LLM call that never produced anything.

(Minor, same area: `round(llm_latency_ms, 2) if llm_latency_ms else None` is a truthiness test, so a genuine `0.0` becomes `None`. The blocking engine correctly uses `is not None`.)

### 7b. The streaming engine never writes the final trace attributes

The blocking engine closes its trace with:

```python
# execution_engine.py:254
tracing.set_trace_attributes(
    llm_ran=llm_ran,
    output_validation=(agent_decisions.get("output_validation") or {}).get("is_valid"),
)
```

The streaming engine has no equivalent. Since the SSE engine is the one serving production, **every production Langfuse trace is missing `llm_ran` and `output_validation`** — they only appear in traces from the blocking engine, which nothing in production calls.

### 7c. The refusal strings differ by one character

```python
# execution_engine.py:250       "I don’t have enough reliable information "   ← U+2019
# execution_engine_streaming.py:389  "I don't have enough reliable information "   ← U+0027
```

Anything that string-matches refusals across both engines — `evaluation/refusal_metrics.py`, `tests/regression_suite.py`, any future guardrail — will classify one engine's refusals and miss the other's.

### Fix

Extract the shared tail (KPI aggregation, refusal constants, trace attributes) into one module both engines import, or make `RageEngine` a thin wrapper over `StreamingRageEngine` with a no-op token callback. Two hand-maintained copies of a 100-line KPI block will keep drifting.

---

## §8 — `candidate_spans()` discards the first token of every query

**Severity:** Medium — can produce a hard refusal for a game that *is* in the corpus.
**File:** `retriever/corpus_index.py:164-166`

### What the code does

```python
for idx, tok in enumerate(tokens):
    if idx == 0:
        continue
```

### What was intended

The docstring is honest about it: *"The sentence-initial token is never considered (golden-set queries are always interrogative: 'What...', 'Which...', 'Can I...') so it never seeds a span."* Against the golden set, that is a correct and cheap way to stop `What` from seeding a span.

### What is actually wrong

Real users don't type golden-set queries. `Far Cry 5 combat` tokenizes to `["Far", "Cry", "5", "combat"]`; `Far` is dropped, so the span is `("cry", "5")`, which is not in `known_titles`. The only thing standing between that and a refusal is the `source_title` substring fallback:

```python
if span_text and span_text in source_titles:
    return True
```

That works when retrieval happened to return a chunk titled e.g. *"Far Cry 5 Review"* — `"cry 5"` is a substring. It fails when the retrieved chunks are titled differently (a franchise piece, a hardware article, a roundup). Then `assess_grounding` returns `False`, and `quality_gate.evaluate()` takes the hard path:

```python
status=QualityStatus.QUALITY_EMPTY,
reason="Query entity absent from corpus",
```

`QUALITY_EMPTY` → `CapabilityAssessor` returns `INSUFFICIENT` → the LLM is bypassed entirely → the user is told the system has no reliable information about a game that is fully ingested.

This is currently the *only* live refusal path (see §1), which makes it the sole thing standing between the system and answering everything — so its false positives matter more than they would otherwise.

### Fix

Only skip index 0 when the first token is an actual interrogative. A small set (`what which who when where why how can does is are do`) covers it, and `_STOPWORDS` already contains most of them:

```python
if idx == 0 and tokens[0].lower() in _STOPWORDS:
    continue
```

Add a regression test for a bare-title query — `"Far Cry 5 combat"` — asserting `assess_grounding` is not `False`.

### Resolved — 2026-08-19

Shipped a different fix than the snippet proposed above. That snippet —
`if idx == 0 and tokens[0].lower() in _STOPWORDS: continue` — was hand-traced
against `_TOKEN_RE` (which keeps apostrophes) and found to introduce new
false refusals of its own: `"What's the best co-op shooter?"` → `"what's"`
is not in `_STOPWORDS` → seeds a junk span → refuses (**5 of the 50
golden-set queries open with "What's"**); `"Tell me about combat"` →
`"tell"` not in `_STOPWORDS` → refuses where today it's `None`; `"Compare
Far Cry 5 and Doom Eternal"` → `"Compare"` absorbs into the adjacent span →
`("compare","far","cry","5")` → no match → a previously-grounded query
would flip to refused.

Instead, `candidate_spans()` gained an `include_sentence_initial` keyword
(default `False`, preserving today's behavior exactly) and
`assess_grounding()` now checks two span sets: the conservative default
(index 0 excluded) as the "does this query name any entity at all" verdict,
plus the greedy `include_sentence_initial=True` variant for matching. Both
sets are checked against `known_titles`/`source_titles` — not just the
greedy one — because including index 0 can merge a leading non-entity word
into what would otherwise be a clean span (the `"Compare Far Cry 5..."`
case above), and the conservative set already splits that correctly. This
construction is monotone: it can only turn a `False` into a `True`, never
`None`/`True` into `False`.

Tests: `tests/test_corpus_index.py` gained 6 new cases covering the bare-title
regression this section asked for, the `"Compare X and Y"` and `"What's..."`
non-regressions, and the sentence-initial-title recovery case, alongside all
8 pre-existing tests (unchanged, still passing — including
`test_candidate_spans_ignores_sentence_initial_token`, whose comment was
reworded since it's now only true at the span level, not the grounding
level).

---

## §9 — The corpus entity index never refreshes

**Severity:** Low — stale after ingestion.
**File:** `retriever/corpus_index.py:239-245`

```python
_entity_index: Optional[CorpusEntityIndex] = None

def _get_entity_index() -> CorpusEntityIndex:
    global _entity_index
    if _entity_index is None:
        _entity_index = CorpusEntityIndex()
    return _entity_index
```

Loaded once per process, from a single Qdrant scroll of the `Game` collection. Any game ingested after the API booted is not in `known_titles`, so queries about it hit §8's `QUALITY_EMPTY` refusal path until the service restarts.

On Render this partly hides itself — the free tier spins down after ~15 minutes idle, so the index is rebuilt often. That is luck, not design. Add a TTL, or an explicit invalidation the ingest path can call.

### Resolved — 2026-08-19

Added a TTL (`CORPUS_INDEX_TTL_SECONDS`, default 900s, `<= 0` disables
refresh) plus a `threading.Lock`-guarded single-flight rebuild: on expiry,
one thread refreshes while concurrent requests keep serving the current
index rather than queueing behind a Qdrant scroll (the API spawns a thread
per request — see `api/main.py`). The TTL is read from `os.environ` per
call rather than cached, matching the existing convention in
`quality_gate._resolve_floors()`.

Two fail-soft details worth calling out: `_entity_index_loaded_at` is
bumped *before* the rebuild attempt, so a persistently-failing Qdrant
retries once per TTL window rather than on every request during the
outage; and a refresh that comes back with an empty `known_titles` (the
constructor's existing behavior on a swallowed load exception) keeps the
previous good index instead of replacing it — an empty index would make
every query un-groundable.

Also added `invalidate_entity_index()` as an explicit seam, but **not**
wired to the ingest path as this section originally suggested:
`scripts/bulk_ingest.py` and `upsert/*` run as separate CLI processes and
cannot reach this module's in-process state, so a hook there would do
nothing for the real staleness. The TTL is the actual fix for that path;
the invalidation function exists for tests and any future in-process
ingest trigger.

Tests: `tests/test_corpus_index_ttl.py` (new, 5 tests) — caching within
TTL, rebuild on expiry, keep-previous-index on a failed/empty refresh,
refresh disabled at `TTL<=0`, and `invalidate_entity_index()` forcing a
rebuild even with refresh disabled.

---

# Part B — Built, then never connected

Working code that nothing calls, or fields nothing reads.

---

## §10 — Three `RouterDecision` fields are computed and never read

**Severity:** Low (dead code) / Medium (misleading contract).
**Files:** `agent/task_router.py:102-104`; `retriever/strategy_selector.py:59-60`

`TaskRouter.route()` populates:

```python
retrieval_strategy=self._retrieval_strategy(task),
web_search_allowed=self._web_allowed(task),
max_results=self._max_results(task),
```

`StrategySelector.select()`, the only consumer of a `RouterDecision`, reads exactly two fields:

```python
task = decision.task                                   # :59
intent_signals: Set[IntentSignal] = decision.intent_signals   # :60
```

and then re-derives limits and the web-fallback flag from `task` in its own `if` ladder. The router's three helper methods (`_retrieval_strategy`, `_web_allowed`, `_max_results`) are pure overhead, and the decision object advertises a routing contract that nothing honours.

Worse than dead code: the two derivations can disagree without anything failing. `_max_results` returns `10` for LISTICLE and `5` otherwise; `StrategySelector` independently returns `limit=10` for LISTICLE, `7` for mixed-intent FACTUAL, `5` otherwise. The `7` has no counterpart in the router. Anyone reading `RouterDecision.max_results` to understand retrieval breadth gets the wrong answer.

**Fix:** pick one owner. Either `StrategySelector` consumes the router's fields, or the router stops computing them.

### Resolved — 2026-08-19

Fixed alongside §11 (they're causally coupled — see the status update at the top of this file). Shipped the "delete" half of the fix rather than "consume": `retrieval_strategy`, `web_search_allowed`, and `max_results` are removed from `RouterDecision` entirely, along with `TaskRouter`'s `_retrieval_strategy()`, `_web_allowed()`, `_max_results()` helper methods and their call sites in `route()`. `RouterDecision` now carries only `task`, `intent_signals`, and `reason` — the fields `StrategySelector.select()` actually reads. A repo-wide grep confirmed the only other references to the three removed fields were `strategy_selector.py`'s own `if __name__` test-harness literals, updated in the same pass; `agent_decisions["retrieval_strategy"]` in both engines was already built entirely from `StrategySelector`'s output, never from the router's fields, so this changes zero observable behavior.

Tests: `tests/test_strategy_selector.py` (new, 11 tests) — see §11's Resolved note, since the same file covers both fixes.

---

## §11 — Web fallback is only reachable for `TaskType.OPEN`

**Severity:** Medium — the temporal-freshness path is unreachable for the queries that need it most.
**File:** `retriever/strategy_selector.py:70, 81, 94, 101, 111`

Of five `RetrievalConfiguration` constructions, exactly one sets `allow_web_fallback=True` — the `OPEN` fallback branch (line 111). COMPARISON, LISTICLE, and both FACTUAL variants are all `False`.

`orchestrator.py` gates the temporal trigger on it:

```python
elif (
    quality_report.status == QualityStatus.QUALITY_WEAK
    or (config.allow_web_fallback and quality_report.has_temporal_signal)
):
```

So *"latest patch notes for Assassin's Creed Valhalla"* — which `IntentSignalExtractor` tags `FACTUAL` + `TEMPORAL`, routing to `TaskType.FACTUAL` — can never trigger a web search on the temporal path, no matter how stale the corpus is. Only a genuinely `OPEN` query (no comparison, no listicle, no factual marker) can.

Stack this with §1 and the picture in production is:

- `QUALITY_WEAK` never fires (§1) → the first half of the `elif` is dead
- `allow_web_fallback` is `False` for most tasks (§11) → the second half is dead for them
- **Net: web search fires only on `QUALITY_EMPTY`** — that is, only when retrieval returned literally nothing, or when the entity-grounding check refused (§8).

`WebSearchTool`, `decide_web_search()`, `_refine_web_results()`, the web re-gate, and the `LOCAL_PLUS_WEB` merge state are all built, tested, and essentially unreachable in the current configuration.

**Fix:** decide deliberately which tasks may reach the web. A TEMPORAL signal is a strong argument for allowing it regardless of task type — freshness is orthogonal to whether the question is factual or a comparison.

### Resolved — 2026-08-19

Fixed alongside §10 (see the status update at the top of this file). Shipped exactly the fix this section recommended: `StrategySelector.select()` now computes `has_temporal = IntentSignal.TEMPORAL in intent_signals` once at the top of the method, and the COMPARISON, LISTICLE, and both FACTUAL branches set `allow_web_fallback=has_temporal` instead of a hardcoded `False`. The OPEN branch's `allow_web_fallback=True` stays unconditional — orthogonal, exploratory-fallback behavior, not something a temporal check should narrow. `orchestrator.py`'s existing gate (`config.allow_web_fallback and quality_report.has_temporal_signal`) required no change; it already read the field correctly; it just could not observe `True` from a non-OPEN task before this fix.

Deliberately left alone: `quality_report.has_temporal_signal` (`retriever/quality_gate.py`) checks whether *retrieved chunk content* contains temporal language — a distinct signal from `IntentSignal.TEMPORAL`, which is extracted from the *query* by `agent/intent/intent_extractor.py`. Conflating the two would have been an easy mistake; this fix only removes the task-type gate blocking the content-level check from ever running for non-OPEN tasks.

Tests: `tests/test_strategy_selector.py` (new — no test file existed for this module before). 11 tests: for each of COMPARISON, LISTICLE, FACTUAL (single- and mixed-intent), `allow_web_fallback` is `False` without `TEMPORAL` and `True` with it; OPEN stays `True` regardless; plus a regression test matching this section's own example (`FACTUAL` + `TEMPORAL` → `allow_web_fallback=True`).

---

## §12 — `insufficient_prompt()` is constructed and discarded

**Severity:** Low.
**Files:** `agent/prompt_templates.py:223`; `agent/prompt_manager.py:82`; both engines' STEP 7

`PromptManager.generate_prompt()` handles the refusal case properly:

```python
if capability == AnswerCapability.INSUFFICIENT:
    registry.record("prompt_mode", "insufficient")
    registry.record("prompt_budget_mode", "insufficient_safe_refusal")
    return insufficient_prompt(query)
```

Both engines then throw it away:

```python
if capability != AnswerCapability.INSUFFICIENT:
    ...generate...
else:
    final_answer = "I don't have enough reliable information to answer this request safely."
```

The prompt is built, two metrics are recorded about it, and it is never sent anywhere. Users get a fixed string instead of the honest, query-aware refusal the template was written to produce.

**Fix:** either send it (a refusal that names what was asked and why the evidence fell short is materially better UX, at the cost of one LLM call on the refusal path), or delete `insufficient_prompt()` and the two metric records. The current state is the worst of both — the cost of maintaining it, none of the benefit.

---

## §13 — The entire `stage` event stream is ignored by the UI

**Severity:** Medium — real UX cost on the free tier.
**Files:** `engine/execution_engine_streaming.py` (`emit_stage`, 7 call sites); `api/main.py:110-118`; `frontend/src/app/page.tsx:434`

The backend emits a `stage` SSE event at the start and end of all 7 pipeline steps, each carrying `name`, `status`, `duration_ms`, and a `data` payload (task type, intent signals, chunk counts, merge state, quality status, prompt length, validation result). `api/main.py` serialises and forwards every one.

The frontend:

```javascript
// 'stage' events are progress only -- nothing to render yet.
```

Nothing. Not stored, not displayed.

**Why this matters concretely:** CLAUDE.md records that retrieval + rerank measures **106–122 seconds** on Render's throttled free-tier CPU. For that entire time the user sees a three-dot bouncing `TypingIndicator` and no indication that anything is happening, whether it is stuck, or how far along it is. The information needed to fix that is already arriving over the wire, fully structured, and is being discarded.

Also unrendered: `agent_decisions` in the `done` payload — the quality report, the web-search decision with its reason and confidence, the routing reason, and the output-validation result. All computed, all serialised, all dropped.

**Fix:** render the stage stream as a progress list with per-stage timings. This is the single highest-value UI change available and requires no backend work.

---

## §14 — No conversation history

**Severity:** Medium — product-shaped gap, not a bug.
**Files:** `api/main.py:71` (`class ChatRequest`); `frontend/src/app/page.tsx:368`

```python
class ChatRequest(BaseModel):
    query: str
```

```javascript
body: JSON.stringify({ query: userMsg }),
```

The UI is a chat interface — message bubbles, avatars, scrollback, a streaming assistant turn. The API behind it is strictly single-turn. `messages` state exists purely for display; nothing before the current turn is ever transmitted.

A user who asks *"Tell me about Far Cry 5"* and follows with *"what about its story?"* sends a bare `"what about its story?"` into a system with no idea what "its" refers to. Intent extraction sees no signals, routing lands on `OPEN`, retrieval embeds a pronoun, and the honesty gate is left to catch the wreckage — which, per §1, it currently cannot.

**Fix:** either send the last N turns and add a query-rewriting step before intent extraction, or change the UI so it doesn't promise a conversation. Both are defensible; the current mismatch is not.

---

## §15 — `format_llama3_prompt()` is Modal-era leftover

**Severity:** Trivial.
**File:** `retriever/rag_retriever.py:435`

Emits Llama-3 chat-template control tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`). Nothing on the request path calls it — prompt construction lives entirely in `agent/prompt_manager.py` + `agent/prompt_templates.py`. Only `main()`, the module's CLI harness, uses it, and that harness sends the result to Gemini/Groq, neither of which uses Llama-3 control tokens. CLAUDE.md states there is no Modal dependency anywhere; this is its last residue.

**Fix:** delete it, and have the CLI harness call `PromptManager` so it exercises the real path.

---

# Part C — Configuration, drift, and hygiene

---

## §16 — `fastembed` is unpinned in the root requirements

**Severity:** Medium — a future rebuild can silently invalidate the calibrated floors.
**Files:** `requirements.txt:12`; `hf_space/requirements.txt`

```
# requirements.txt:12
fastembed                       # BM25 sparse vector generation + in-process cross-encoder reranking

# hf_space/requirements.txt
# fastembed is pinned to the exact version RAGent runs locally (0.7.4).
fastembed==0.7.4
```

The Space pins itself *to the root*, and the root is unpinned. That inverts the dependency: the invariant is documented in the file that cannot enforce it.

Two things break if a Render rebuild pulls a newer `fastembed`:

1. `_FLOORS["hfspace"] = (-3.0, 2.0)` is justified entirely by the claim that the Space's scores are bit-identical to local's (verified to 0.000000000). Different version, different ONNX graph or tokenizer, claim gone — and CLAUDE.md says that entry must revert to `None` if it ever drifts. Nothing would detect the drift.
2. `_FLOORS["local"]` is calibrated against a specific model build. Same exposure.

Also relevant: `Dockerfile` bakes the models into the image at build time via `FASTEMBED_CACHE_PATH`, so a version bump changes what gets baked, silently, on the next deploy.

**Fix:** `fastembed==0.7.4` in the root requirements. Bump both files together, deliberately, and re-calibrate when you do.

---

## §17 — Every stored evaluation result predates the current system

**Severity:** Medium — the published numbers describe a system that no longer exists.
**Directory:** `evaluation/results/`

| Artifact | Date | Measured against |
|---|---|---|
| `ragas_2026-08-09_default.json` | 08-09 | E5 embeddings, Modal reranker, retired models |
| `ablation_2026-08-09.json` | 08-09 | same |
| `refusal_2026-08-12_*.json` | 08-12 | E5 embeddings, local reranker |
| `relevance_calibration_2026-08-12.json` | 08-12 | E5 embeddings, local reranker |
| `cost_latency_2026-08-14_default.json` | 08-14 | pre-model-swap pricing |

Since 08-14 the system has changed: dense embeddings moved E5 → `gemini-embedding-001` (§3), the reranker moved to Cloudflare `bge-reranker-base`, `gemini-2.5-flash` → `gemini-flash-lite-latest`, and Groq `llama-3.1-8b-instant` → `openai/gpt-oss-120b`.

The most consequential item is `relevance_calibration_2026-08-12.json`. It is the **sole justification** for `_FLOORS["local"] = (-3.0, 2.0)`, and it was measured on the pre-migration E5 corpus. The floors threshold the *cross-encoder's* output, and the cross-encoder scores whatever the retriever hands it — so changing which chunks retrieval surfaces changes the score distribution those floors were fitted to. Even reverting `RERANKER_PROVIDER=local` today would not restore a correctly-calibrated gate.

**Fix:** after §3 completes, re-run in order: `calibrate_relevance` (→ §1), then `refusal_metrics`, `ragas_eval`, `ablation`, `cost_latency`. Keep the old files; the date-stamped naming already supports side-by-side comparison, and the before/after is genuinely interesting.

---

## §18 — `is_noise()` discards ordinary review prose

**Severity:** Medium — silent evidence loss ahead of the gate.
**File:** `retriever/quality_gate.py:72-77, 295-298`

```python
NOISE_KEYWORDS: Set[str] = {
    "sale", "sales", "discount", "deal", "bundle", "price", "store",
    "buy", "purchase",
    "community", "forum", "thread", "discussion",
}

def is_noise(self, title: str, content: str) -> bool:
    text = f"{title} {content}".lower()
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in self.NOISE_KEYWORDS)
```

Word-boundary match against the **full concatenated chunk content**, not the title, not a density measure. Any single occurrence anywhere in ~1500 characters discards the whole chunk.

Game-review prose that trips this:

- "a great **deal** of freedom" / "the best **deal** in the genre"
- "the **price** of failure is steep" / "at what **price**?"
- "the in-game **store**" — describing a mechanic, not selling anything
- "the modding **community**", "the speedrunning **community**"
- "a lengthy **discussion** between the two characters"
- "**buy**-in", "the player must **purchase** upgrades" — core progression systems

The keywords were chosen to reject *storefront and forum pages* — a sound goal for web results, where `_refine_web_results()` also calls `is_noise()`. Applying the same rule to local editorial chunks, which are curated GameSpot articles, throws away good evidence before the gate ever scores it. And it happens before `valid_chunks` is built, so the drop is invisible: `evidence_count` reports the survivors, with no counter for what was removed.

**Fix:** scope it. Match on `source_title` and URL for the commerce/forum terms, keep a much narrower keyword set for content, or require a density threshold (several distinct hits, not one). At minimum add `MetricsRegistry.inc("chunks_dropped_as_noise")` so the loss is measurable before you decide.

---

## §19 — A disconnected client leaves the engine running

**Severity:** Low — resource waste on a 512MB / throttled-CPU tier.
**File:** `api/main.py:154-165`

```python
threading.Thread(target=run_engine, daemon=True).start()

async def event_generator():
    while True:
        if await request.is_disconnected():
            break
        item = await asyncio.get_event_loop().run_in_executor(None, q.get)
```

The disconnect check only runs *between* queue items. Once execution reaches the blocking `q.get`, it stays there until the engine produces something. Given the 106–122s retrieval stage, a user who closes the tab during retrieval leaves:

- the engine thread running the full pipeline, including the paid Cloudflare rerank call and the LLM generation, with nobody to receive them
- a thread-pool worker parked on `q.get` for the duration
- `daemon=True`, so it is never joined and never cancelled

Under any real concurrency on a free-tier container, that compounds.

**Fix:** give `q.get` a timeout and re-check `is_disconnected()` each iteration; set a cancellation flag the engine can observe between stages. Full mid-stage cancellation is harder and probably not worth it — stopping at stage boundaries would recover most of the waste.

---

## §20 — `_rerank()` can produce a half-scored candidate list

**Severity:** Low — narrow, but it corrupts the gate's input when it fires.
**File:** `retriever/rag_retriever.py:394-405`

```python
try:
    contents = [c.get("content") or "" for c in candidates]
    rerank_scores = _rerank_scores(query, contents)

    for c, s in zip(candidates, rerank_scores):     # ← zip truncates silently
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)   # ← KeyError
except Exception as exc:
    logger.warning(f"Reranker unavailable (fail-soft): {exc}")

return candidates[:limit]
```

If a provider returns fewer scores than candidates, `zip` stops at the shorter sequence with no error. The first N candidates get a `rerank_score`; the rest do not. The `sort` then raises `KeyError` on the first unscored one. The `except` catches it and preserves RRF order — but the mutation already happened, so the returned list is **partially scored**.

Downstream, `quality_gate.evaluate()` collects only the non-`None` `rerank_score` values and takes `max(rerank_scores)` over that partial set. The gate grades on a subset of the evidence while believing it graded all of it. And §2's fix — sorting context by `rerank_score` with a fallback — would sort a mixed list across two incompatible scales.

`llm/cloudflare_rerank_client._parse_scores()` explicitly guards against this (it raises rather than padding a short response, per its documented rationale). The `local`, `hfspace`, and `voyage` paths do not.

**Fix:** length-check before mutating:

```python
rerank_scores = _rerank_scores(query, contents)
if len(rerank_scores) != len(candidates):
    raise ValueError(f"reranker returned {len(rerank_scores)} scores for {len(candidates)} candidates")
```

Placed before the loop, this turns a partial mutation into a clean fail-soft fallback to RRF order.

### Resolved — 2026-08-19

Fixed alongside §2. Two corrections to this section's own analysis, made while implementing:

1. **The guard moved to `_rerank_scores()`, not `_rerank()`.** `score_relevance()` has the identical exposure — its result is zipped onto `web_chunks` at `orchestrator.py:229` — and `_rerank_scores()` is the single dispatch point both callers already share (its own docstring says so). A length check inside `_rerank()` alone would have left `score_relevance()`'s web-evidence path unguarded.
2. **The provider table above has two errors.** Checked each client directly: `llm/hf_rerank_client.py:131-134` **does** raise on a length mismatch — `hfspace` was already guarded, contrary to what's stated above. `local` — the default provider, and the one this whole finding is really about — calls `reranker.rerank()` (a bare generator) with no check at all; it's the actually-unguarded path. `voyage` can't return short (it pre-allocates `[0.0] * len(documents)`), but silently pads with `0.0` for omitted documents instead, which is arguably worse on a 0..1 scale since it reads as "confirmed irrelevant" rather than "unscored."

`_rerank_scores()` now raises `ValueError` on any length mismatch, for every provider branch, before returning. `_rerank()` was also restructured so scoring and mutation are separate phases — nothing is written to any candidate until every score is validated — since a length check alone doesn't prevent the *original* code's problem: it mutated candidates as it scored them, inside the same `try` the check would live in, so a *later* failure could still leave earlier candidates partially mutated. `score_relevance()` needed no code change; its existing `except` already converts the new `ValueError` into its documented all-`None` fail-soft return.

Tests: `tests/test_rerank_failsoft.py` gained 3 tests (up from 4) — one exercising `_rerank()`'s own inner length check, two exercising `_rerank_scores()`'s guard directly (via a stub reranker returning one fewer score than requested, rather than monkeypatching `_rerank_scores` itself, which would bypass the exact code under test).

---

## §21 — Comments and docs describe retired infrastructure

**Severity:** Trivial individually; collectively a real hazard, since these files are what the next reader trusts.

| Location | Says | Reality |
|---|---|---|
| `requirements.txt:3-6` | "LLM: Gemini 2.5 Flash (primary) + Groq Llama 3.1 8B Instant (fallback)" | `gemini-2.5-flash` 404s for this account; `llama-3.1-8b-instant` is retired. Both replaced. |
| `vector/create_schema.py:39` | `E5_VECTOR_SIZE = 768` | Gemini `gemini-embedding-001` dimensionality. Name is a lie; the value is right. |
| `vector/create_schema.py:50` | `"dense": E5-base-v2 (768-dim, cosine)` in the docstring | Same. |
| `CLAUDE.md` | multi-source ingestion is "RAWG, IGDB, GameSpot" | Wikipedia and Steam editorial are also wired in — `upsert/upsert_all.py:32`, `scripts/bulk_ingest.py:48`, `ingest/editorial_providers.py`, `ingest/wikipedia_editorial_normalize.py`, `ingest/steam_editorial_normalize.py`, `upsert/upsert_editorial_source.py`. |
| `retriever/rag_retriever.py` module docstring | "the in-process fastembed cross-encoder, or Voyage's rerank HTTP API" | Four providers now; Cloudflare is the active one. |

---

## §22 — `transformers` is installed and never imported

**Severity:** Low — dead weight in a 512MB image.
**File:** `requirements.txt:22`

```
transformers                    # Tokenizers for chunking
```

Nothing in the repo imports it — verified across every `.py` file. `chunking/editorial_chunker.py` ships its own `LocalTokenizer` that splits on whitespace:

```python
class LocalTokenizer:
    """Splits on ANY whitespace (spaces, tabs, newlines, unicode)"""
```

Two consequences:

1. It is pure image weight on a tier where CLAUDE.md warns that RAM is "genuinely tight." It is not imported on the request path, so it does not consume RSS — but it inflates build time and image size for nothing.
2. `chunk_size=500` means **500 whitespace-delimited words**, not 500 model tokens. The measured corpus confirms it: median 1507 characters ≈ 250–300 real tokens, roughly half the nominal figure. Any budgeting reasoning that assumed 500-token chunks was working from a number twice the truth.

**Fix:** remove the dependency. Separately, decide whether `chunk_size` should mean tokens — if so, it needs a real tokenizer and re-chunking; if not, rename the parameter to `chunk_words` so nobody re-derives the wrong budget from it.

---

## §23 — Two passing tests lock the broken state in place

**Severity:** Meta, but important.
**File:** `tests/test_llm_config.py:85, 96`

```python
def test_voyage_floors_are_uncalibrated_placeholder():
def test_cloudflare_floors_are_uncalibrated_placeholder():
```

These assert that `_FLOORS["cloudflare"] is None` and `_FLOORS["voyage"] is None`.

They were correct when written: they encode "never copy another provider's floors onto a different score scale," which is a genuinely important invariant and exactly the kind of mistake worth a regression test.

But the effect today is that the suite goes green while **asserting that the production reranker has no calibrated floors** — i.e. while asserting §1. The single most consequential defect in the codebase is not merely undetected by the tests; it is pinned by them. Nothing will ever warn you.

**Fix:** when §1 lands, invert `test_cloudflare_floors_are_uncalibrated_placeholder` into an assertion that the floors exist, are a 2-tuple, are ordered `refuse < weak`, and sit inside the provider's actual score range (0..1 for Cloudflare). Keep the Voyage test as-is until Voyage is calibrated. Consider adding a meta-test that fails whenever `RERANKER_PROVIDER` names a provider whose `_FLOORS` entry is `None` — that turns "the gate is off" from an invisible state into a red build.

---

# Summary

**Nothing is crashing.** Every finding here is code that runs, returns, and reports success while not doing its job.

The three that change what users receive:

1. **§1 — the honesty gate is off.** An uncalibrated `None` disables the relevance ladder for the active reranker, so every query with any evidence is graded `QUALITY_OK`, PARTIAL is nearly unreachable, and the weak-evidence web-search path is dead. Tooling to fix it is written and has never been run.
2. **§2 — reranking is thrown away.** *(Fixed 2026-08-19, together with §20 — see status update at the top of this file.)* Context assembly re-sorts by the RRF score, then a 4000-char budget admits 2–3 of the ~1500-char chunks. The cross-encoder pays full cost and barely influences the prompt.
3. **§3 — the migration is 91 chunks short.** Those chunks are being searched with mismatched embedding spaces right now.

**Order of operations matters:** §3 → §1 → §17. Calibrating or evaluating before the corpus is uniform bakes the mismatch into the floors and into every published number.

The recurring theme in Part B is a pipeline that computes more than it consumes — routing fields nobody reads, prompts built and discarded, stage events streamed and ignored, an agentic decision that cannot be reached. Each is individually small. Together they mean the system's observable behaviour is substantially simpler than its architecture implies, and the difference is not visible from the outside.
