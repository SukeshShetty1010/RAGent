# RAGent — Audit & Task List

**Original audit:** 2026-08-18 — every module on the request path (`api/` → `engine/` → `agent/` → `retriever/` → `llm/` → `utils/` → `frontend/`), plus ingest, evaluation, config, Docker, and CI, read end to end. Claims about live behaviour are measured against the real Qdrant corpus and the real `.env`, never inferred.

**Status (2026-08-23):** T1–T25, T28, T33 closed. T26, T27, T29–T32 open, raised while finishing T17. Test suite `285 passed, 3 skipped` full run including `live`-marked tests; `274 passed, 3 skipped` on the hermetic `-m unit` subset.

**How this file is kept:** a task collapses to one line in the Closed table as soon as its fix ships, because the durable knowledge from each fix lives in the code comments and regression tests that fix added — not here. The full per-task analysis and the dated status updates for T1–T23 remain in git history (`git log -p -- AUDIT_TASKS.md`). Open tasks keep their full evidence until they are fixed.

---

## Open tasks

- [ ] **T26** — Measure the real-corpus noise drop rate that §18 left unrun (§26)
- [ ] **T27** — `flagship.md` publishes superseded evaluation numbers (§27)
- [ ] **T29** — `requirements-dev.txt` is unpinned and does not install cleanly (§29)
- [ ] **T30** — The `KPI/` suite has never been re-run against the migrated corpus (§30)
- [ ] **T31** — `ablation.py --limit` overwrites the real results file (§31)
- [ ] **T32** — `--rescore-modes` can only target today's results file (§32)

---


## §26 — The real-corpus noise drop rate §18 called for was never measured

**Severity:** Medium — an unverified assumption sitting under a live filter.
**File:** `retriever/quality_gate.py` (`SOURCE_NOISE_KEYWORDS`, `is_noise()`)

T18 narrowed the noise filter and shipped 10 regression tests, but its Resolved note explicitly recorded one thing as not done: the old-rule-vs-new-rule drop rate against the real corpus, by keyword. It was blocked at the time because every Qdrant attempt timed out from that environment.

**That blocker is gone.** The timeouts were the `QDRANT_URL` port issue (the client appends 6333; some networks allow only 443), not an absent corpus — Qdrant now connects in 0.86s from this machine. The measurement is runnable and remains unrun, so `MetricsRegistry.inc("chunks_dropped_as_noise")` still has no baseline to be read against.

**Fix:** scroll the `EditorialChunk` collection, apply both the pre-T18 and post-T18 rules to every chunk, and report the drop rate for each plus a per-keyword breakdown. Store it as a dated artifact under `evaluation/results/`.

---

## §27 — `flagship.md` publishes superseded evaluation numbers

**Severity:** Medium — this is the most public-facing document in the repo.
**File:** `flagship.md:255-281`, `:484-488`

It carries the 2026-08-09 ablation table verbatim (`dense | 0.9400 | 0.8875 | 0.6537`, and the claim "RRF hybrid beats dense/BM25 on precision@k (0.95 vs 0.94/0.935)") plus RAGAS `context_precision 0.5722`, `faithfulness 0.9077`. Every one of those figures was superseded on 2026-08-23, and the headline claim actually inverted: `dense` now leads precision@k at 0.9850, and `hybrid_rerank` does not win.

T21 ruled `flagship.md` out of scope as a "historical design record". That was defensible for a stale infrastructure reference; it is weaker for published metrics, since T17's whole premise is that stale numbers mislead whoever reads them next.

**Fix:** a decision, not a mechanical edit — either update the tables to the 08-23 measurements, or stamp the section as a dated snapshot with a pointer to `evaluation/results/`. Note that `context_precision` cannot be compared across the two dates (different judges — see Facts below), so a naive column swap would be its own distortion.

---

## §29 — `requirements-dev.txt` is unpinned and does not install cleanly

**Severity:** Low — but it re-breaks for the next person on a fresh machine.
**File:** `requirements-dev.txt`

The file lists `ragas` unpinned and does not mention `langchain-community` at all. Latest `ragas` (0.4.3) unconditionally imports `langchain_community.chat_models.vertexai`, a module `langchain-community` removed in its 0.4.x split — so the newest versions of both, which is what unpinned `pip install` resolves to, are mutually incompatible.

This was diagnosed on 2026-08-21 and fixed by pinning `langchain-community==0.3.27` **in that session's environment only**; the file was deliberately left alone. The result is that the knowledge lives in prose while the file that would prevent the breakage still reproduces it.

**Fix:** add the `langchain-community<0.4` constraint (and consider pinning `ragas`) with a comment naming the vertexai import as the reason. Dev-only dependency chain, never imported on the Render request path, so pinning it costs nothing at runtime.

---

## §30 — The `KPI/` suite has never been re-run against the migrated corpus

**Severity:** Unknown, and that is the finding.
**Directory:** `KPI/` (`Unified_KPI_Runner.py` plus 5 modules)

T17 scoped `evaluation/` only. `KPI/Unified_KPI_Runner.py` orchestrates a second, independent measurement surface — grounding fidelity, honesty rate, routing accuracy, retrieval quality, latency attribution — and it has no stored outputs anywhere in the repo, so there is no way to tell whether it still runs at all after the embedding migration, the reranker swap, the model swaps, and the engine reconciliation (T7).

**Fix:** run `python -m KPI.Unified_KPI_Runner` and find out. If it errors, that is a real defect the evaluation re-run did not cover; if it succeeds, persist the output as a dated artifact so this question is answerable next time without re-running it.

---

## §31 — `ablation.py --limit` overwrites the real results file

**Severity:** Low — a foot-gun that has already fired once.
**File:** `evaluation/ablation.py`

`--limit N` exists for smoke tests, but the output path is derived only from today's date, so a 2-query smoke run writes to the same `ablation_<date>.json` a real 40-query run does. During T17 a smoke run clobbered the real file and had to be deleted by hand; the only thing that made that recoverable was noticing `"n": 2` in the output.

**Fix:** when `--limit` is set, write to a clearly-marked path (e.g. `ablation_<date>_smoke.json`), or refuse to overwrite a file whose `n` is larger than the current run's.

---

## §32 — `--rescore-modes` can only target today's results file

**Severity:** Trivial.
**File:** `evaluation/ablation.py`

`--rescore-modes` merges into `ablation_<today>.json`. A repair that crosses midnight — likely, since the thing it repairs is usually a daily-quota casualty and the quota resets at 07:00 UTC — cannot reach yesterday's file, and the run errors out rather than merging.

**Fix:** a `--results-file PATH` override for both the read and the write.

---

## Closed — T1–T23

Full analysis in git history (`git log -p -- AUDIT_TASKS.md`); the durable constraints from each live in code comments and regression tests.

| # | Task | Fixed | Outcome |
|---|---|---|---|
| T1 | Calibrate the Cloudflare reranker floors — the honesty gate was off | 08-21 | `_FLOORS["cloudflare"] = (0.02, 0.90)` from a 50-query calibration run; gate live |
| T2 | Order assembled context by `rerank_score`, not the RRF `score` | 08-19 | All-or-nothing `is_fully_reranked()` / `relevance_key()`, never mixing two scales in one sort |
| T3 | Finish the last 91 chunks of the Gemini embedding migration | 08-21 | Self-detecting `--repair` mode (cosine-to-centroid gap split); 0 outliers remain |
| T4 | Remove the `max_tokens=150` override in `decide_web_search` | 08-20 | Deleted; the reasoning-model default applies |
| T5 | Scope `MetricsRegistry` per request | 08-19 | Per-thread via `threading.local()`; no cross-request KPI contamination |
| T6 | `last_used_model()` must report the model that served the *answer* | 08-19 | Explicit `answer_model` recorded where the answer is produced; surfaced in KPIs and UI |
| T7 | Reconcile the two engines | 08-20 | `RageEngine` is now a thin subclass of the streaming engine — one pipeline body, not two |
| T8 | `candidate_spans()` must handle non-interrogative queries | 08-19 | Monotone verdict/match span-set split; no new false refusals |
| T9 | Entity index must refresh without a process restart | 08-19 | TTL refresh |
| T10 | Consume or delete `RouterDecision`'s three dead fields | 08-19 | Deleted — nothing read them, and the two derivations disagreed |
| T11 | Web fallback reachable outside `TaskType.OPEN` | 08-19 | `allow_web_fallback` now set on TEMPORAL intent for COMPARISON/LISTICLE/FACTUAL |
| T12 | Send `insufficient_prompt()` to the LLM or delete it | 08-20 | Wired into both engines' STEP 7 with one shared refusal constant |
| T13 | Render the `stage` SSE events in the UI | 08-20 | `StageProgress` panel driven by the 7 `emit_stage()` sites |
| T14 | Decide whether the product is multi-turn | 08-20 | STEP 0 query condensation (`agent/decisions/query_rewrite.py`); `history` is a per-call kwarg, never engine state |
| T15 | Delete `format_llama3_prompt()` | 08-21 | Gone; CLI harness now reproduces the production prompt path |
| T16 | Pin `fastembed` in the root `requirements.txt` | 08-21 | Pinned to `0.7.4`, matching `hf_space/`; drift had already occurred undetected |
| T17 | Re-run the evaluation suite — every stored result predated the system | 08-23 | All 5 steps re-run; 4 measurement-layer defects fixed (see Facts) |
| T18 | Narrow `NOISE_KEYWORDS` so ordinary review prose isn't discarded | 08-20 | Source-field matching + ≥3-distinct-hit density rule; drop counter added. Corpus drop-rate still unmeasured → **§26** |
| T19 | Cancel the engine thread when the SSE client disconnects | 08-20 | `cancel_event` checkpoints at the 7 stage boundaries; `q.get(timeout=1.0)` |
| T20 | Harden `_rerank()` against a short score list | 08-19 | Length check in `_rerank_scores()`, the shared dispatch point; scoring and mutation are separate phases |
| T21 | Fix stale comments and docs describing retired infrastructure | 08-21 | 5 claims fixed; 3 pinned by regression tests |
| T22 | Drop the unused `transformers` dependency | 08-21 | Removed; chunker units renamed words-not-tokens, docs corrected (~300 words, not "500 tokens") |
| T23 | Invert the two "uncalibrated placeholder" tests once T1 lands | 08-21 | Inverted for Cloudflare; `test_active_provider_floors_are_calibrated` added (see **T24** for its blind spot) |
| T24 | `local`/`hfspace` relevance floors were still calibrated against the pre-migration E5 corpus | 08-23 | Recalibrated via `evaluation/calibrate_relevance.py` against the fully-migrated corpus (`evaluation/results/relevance_calibration_local_2026-08-23.json`); distribution matched the old run within rounding, so `_FLOORS["local"]`/`["hfspace"]` stayed `(-3.0, 2.0)`, now re-validated rather than merely carried forward. `test_active_provider_floors_are_calibrated` rewritten to check every provider's floors against its recorded calibration artifact (new `_CALIBRATION` dict) and reject one generated before `CORPUS_EMBEDDING_MIGRATION_DATE` — "present but stale" now fails instead of passing |
| T25 | Web augmentation was overturning justified refusals | 08-23 | Source-scoped ceiling in `quality_gate.evaluate()`: web `rerank_score`s can no longer promote a status the corpus-only scores didn't already earn (`min(observed, ceiling)`, new `corpus_max_relevance`/`web_max_relevance` fields). Found a second, independent defect during investigation — see **T33** |
| T28 | No cost/latency artifact for the corpus-only run | 08-23 | `cost_latency_2026-08-23_corpusonly.json` produced against the fresh post-T25/T33 corpus-only run (not the stale pre-fix `08-21` file T28 originally named — a same-session re-run made that the more useful baseline) |
| T33 | `assess_grounding`'s source_title fallback grounded off-corpus entities via raw substring containment | 08-23 | Filed and closed same session as T25's second root cause. Replaced with a token-prefix test against each chunk's own tokenized title (`retriever/corpus_index.py`), stripping the title's leading stopwords first so titles like "It Takes Two"/"The Legend of Zelda..." — which a query span never seeds on — still anchor at position 0. Fixed g050 (`"US"` matching mid-word); g047 turned out to already ground correctly (real corpus Game identity) and is refused by T25's ceiling instead |

---

## Current measured baseline — 2026-08-23

Ablation (`evaluation/results/ablation_2026-08-23.json`, retrieval n=40/mode, judged n=20/mode, 0 dropped samples):

| Mode | Precision@K | Entity coverage | RAGAS ctx precision |
|---|---|---|---|
| `dense` | 0.9850 | 0.8750 | 0.4607 |
| `bm25` | 0.9350 | 0.8375 | 0.2683 |
| `hybrid` | 0.9600 | 0.9125 | 0.3550 |
| `hybrid_rerank` | 0.9650 | 0.8875 | 0.3433 |

Pipeline (`refusal_2026-08-23_*`, `ragas_2026-08-21_default_gemini`, `cost_latency_2026-08-21_default`, `cost_latency_2026-08-23_corpusonly`):

- **Refusal (post-T25/T33 fix, re-measured 08-23):** precision 1.0, recall **1.0**, false_answer_rate **0.0**, over-refusal 0.0 — both default and corpus-only modes, 0 errors, same 50 golden queries. Was precision 1.0 / recall 0.7 (default) / 0.8 (corpus-only) before the fix.
- **RAGAS, Gemini judge, 40 answerable:** `context_precision=0.3604`, `faithfulness=0.9608`, `answer_relevancy=0.5953`. Corpus fingerprint: 100 games, 2791 chunks.
- **Latency:** engine `p50=4079.56ms / p95=12422.77ms / p99=13736.41ms`, LLM `p50=843.75ms`. Retrieval is 79.66% of total (Tavily alone 73.12%; nested spans double-count).
- **Cost:** `$0.00000212`/query mean, `$0.000106` total for 50 queries — 49/50 served free by Gemini, 1 Groq fallback.

---

## Facts worth keeping

Measured live, and not recorded anywhere else in the repo:

- **Gemini's free tier enforces two independent caps**, and a long eval run hits both: **15 requests/minute** and **500 requests/day** (`gemini-3.5-flash-lite`, the model `gemini-flash-lite-latest` resolves to). The daily window resets at **midnight Pacific = 07:00 UTC** — hours, not a day, from when it usually trips. Both surface as HTTP 429; only the message body distinguishes them.
- **ragas has no retry layer of its own.** `ragas.executor` catches the exception and records the sample as `NaN`, so the OpenAI SDK's `max_retries` is the only thing between a throttled call and a silently dropped sample — and a dropped sample shows up as a smaller `n`, not an error. Its `RunConfig(timeout=...)` bounds a sample's *entire* scoring including retry waits, so raising retries without raising the timeout just moves the failure.
- **`context_precision` is not comparable across judge backends.** The 08-09 numbers were scored by the retired Modal judge and the 08-23 numbers by Gemini; all four ablation modes fell by a similar ~0.2, which is a stricter judge, not a retrieval regression. Compare forward from 08-23, never back across a judge change.
- **`QDRANT_URL` with no explicit port makes the client use 6333.** Networks that allow only 443 produce `ResponseHandlingException: timed out`, which reads exactly like an unreachable cluster and cost three sessions before it was diagnosed. Appending `:443` fixes it where 6333 is blocked; this machine reaches 6333 fine (0.86s) and needs no suffix.
- **A Gemini-served query costs exactly `0.0`** — `llm/pricing.py` prices the free tier at zero deliberately, not for a missing table entry. Any aggregation that filters cost by truthiness therefore silently drops every free query; that bug reported a per-query mean ~50× too high until T17 caught it.
- **Working interpreter is `RAG_env\Scripts\python.exe` (3.12).** `py -3.10`, named in older notes, does not exist on this machine.
