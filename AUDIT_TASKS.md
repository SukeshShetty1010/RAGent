# RAGent — Audit & Task List

**Original audit:** 2026-08-18 — every module on the request path (`api/` → `engine/` → `agent/` → `retriever/` → `llm/` → `utils/` → `frontend/`), plus ingest, evaluation, config, Docker, and CI, read end to end. Claims about live behaviour are measured against the real Qdrant corpus and the real `.env`, never inferred.

**Status (2026-08-23):** T1–T35 closed, no open tasks. T30's re-run (`evaluation/run_kpi_suite.py`) surfaced T34; fixing T34 surfaced a second, independent defect filed as T35. Test suite `283 passed, 3 skipped` on the hermetic `-m unit` subset (up from 274 — 6 new regression tests from T34's fix, 3 more from T35's).

**How this file is kept:** a task collapses to one line in the Closed table as soon as its fix ships, because the durable knowledge from each fix lives in the code comments and regression tests that fix added — not here. The full per-task analysis and the dated status updates for T1–T23 remain in git history (`git log -p -- AUDIT_TASKS.md`). Open tasks keep their full evidence until they are fixed.

---

## Open tasks

None.

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
| T18 | Narrow `NOISE_KEYWORDS` so ordinary review prose isn't discarded | 08-20 | Source-field matching + ≥3-distinct-hit density rule; drop counter added. Corpus drop-rate measured later as **T26** |
| T19 | Cancel the engine thread when the SSE client disconnects | 08-20 | `cancel_event` checkpoints at the 7 stage boundaries; `q.get(timeout=1.0)` |
| T20 | Harden `_rerank()` against a short score list | 08-19 | Length check in `_rerank_scores()`, the shared dispatch point; scoring and mutation are separate phases |
| T21 | Fix stale comments and docs describing retired infrastructure | 08-21 | 5 claims fixed; 3 pinned by regression tests |
| T22 | Drop the unused `transformers` dependency | 08-21 | Removed; chunker units renamed words-not-tokens, docs corrected (~300 words, not "500 tokens") |
| T23 | Invert the two "uncalibrated placeholder" tests once T1 lands | 08-21 | Inverted for Cloudflare; `test_active_provider_floors_are_calibrated` added (see **T24** for its blind spot) |
| T24 | `local`/`hfspace` relevance floors were still calibrated against the pre-migration E5 corpus | 08-23 | Recalibrated via `evaluation/calibrate_relevance.py` against the fully-migrated corpus (`evaluation/results/relevance_calibration_local_2026-08-23.json`); distribution matched the old run within rounding, so `_FLOORS["local"]`/`["hfspace"]` stayed `(-3.0, 2.0)`, now re-validated rather than merely carried forward. `test_active_provider_floors_are_calibrated` rewritten to check every provider's floors against its recorded calibration artifact (new `_CALIBRATION` dict) and reject one generated before `CORPUS_EMBEDDING_MIGRATION_DATE` — "present but stale" now fails instead of passing |
| T25 | Web augmentation was overturning justified refusals | 08-23 | Source-scoped ceiling in `quality_gate.evaluate()`: web `rerank_score`s can no longer promote a status the corpus-only scores didn't already earn (`min(observed, ceiling)`, new `corpus_max_relevance`/`web_max_relevance` fields). Found a second, independent defect during investigation — see **T33** |
| T28 | No cost/latency artifact for the corpus-only run | 08-23 | `cost_latency_2026-08-23_corpusonly.json` produced against the fresh post-T25/T33 corpus-only run (not the stale pre-fix `08-21` file T28 originally named — a same-session re-run made that the more useful baseline) |
| T33 | `assess_grounding`'s source_title fallback grounded off-corpus entities via raw substring containment | 08-23 | Filed and closed same session as T25's second root cause. Replaced with a token-prefix test against each chunk's own tokenized title (`retriever/corpus_index.py`), stripping the title's leading stopwords first so titles like "It Takes Two"/"The Legend of Zelda..." — which a query span never seeds on — still anchor at position 0. Fixed g050 (`"US"` matching mid-word); g047 turned out to already ground correctly (real corpus Game identity) and is refused by T25's ceiling instead |
| T30 | `KPI/` suite never re-run against the migrated corpus | 08-23 | New `evaluation/run_kpi_suite.py` (additive, zero edits under `KPI/`) imports all 5 modules directly, captures stdout/timing/exceptions per module into a dated artifact (`evaluation/results/kpi_suite_cloudflare_2026-08-23.json`). All 5 modules completed, no crashes, no Groq-fallback signature. Surfaced a real regression (Regression Guard Coverage 1/3, not 3/3) → filed as **T34** rather than fixed here, per T30's own "run it and find out" scope |
| T26 | Measure the real-corpus noise drop rate that §18 left unrun | 08-23 | New `evaluation/measure_noise_drop_rate.py` (additive, imports the live `RetrievalQualityGate.is_noise()` for production parity) scrolled all 2791 `EditorialChunk` points. Old rule dropped 558 (19.99%); new rule drops 122 (4.37%) — 436 recovered by T18 (15.62% of corpus, confirmed incidental-prose false positives on spot check), 0 new-only url-triggered drops. Sanity assertions (scanned==Qdrant count, partition identities) held; re-run is byte-identical. Artifact: `evaluation/results/noise_drop_rate_2026-08-23.json` |
| T34 | Regression Guard Coverage dropped to 1/3 under the Cloudflare reranker — BUG-003 flipped FULL→INSUFFICIENT | 08-23 | Resolved the audit's own open question: a genuine, **provider-agnostic** entity-index bug, not a reranker-floor mismatch (the entity-grounding short-circuit in `quality_gate.evaluate()` fires before floors are ever read). Root cause: `retriever/corpus_index.py`'s `_TOKEN_RE` only treated the ASCII apostrophe as token-internal, so a curly right-quote (U+2019, as written in `tests/regression_suite.py`'s BUG-003 query and in `KPI/Retrieval_Quality_KPI.py`'s fixtures) tokenized "Assassin's" as two tokens instead of one and never matched the corpus's ASCII-apostrophe title — even with 16 real, on-topic Valhalla chunks as evidence, confirmed live. Fixed via a `_APOSTROPHE_VARIANTS` translation table applied in `_tokenize()`. The matching `RetrievalQualityKPI` symptom (Entity Coverage 0.00%, Evidence Hit: NO) needed a second, independent fix in `tests/evaluation_metrics.py`: the same apostrophe gap in `_normalize()`, plus a distinct bug in `_resolve_entity()`, whose key-priority list checked `retrieval_context` before `source_title` — but `agent/tools/web_search.py` sets `retrieval_context="fallback"` as a merge-state marker (not an entity name) on web-augmented evidence, so every web-sourced chunk in a temporal query resolved to the literal string `"fallback"` and never matched. Both fixes verified live: `RegressionRunner().run()` now passes BUG-002/BUG-003 under both `cloudflare` and `local` (BUG-001 still fails, unrelated — filed as **T35**); `evaluation/results/kpi_suite_cloudflare_2026-08-23.json` re-run shows Regression Guard Coverage 2/3, Evidence Hit Rate 100%, Avg Entity Coverage 100%. 6 new regression tests (`tests/test_corpus_index.py` ×2, `tests/test_evaluation_metrics.py` ×4, new file) |
| T27 | `flagship.md` publishes superseded evaluation numbers | 08-23 | Stamped, not overwritten: `flagship.md:255-273`, `:275-292`, `:520/524` each get a "Superseded — current numbers as of 2026-08-23" block with the `ablation_2026-08-23.json`/`ragas_2026-08-21_default_gemini.json` figures and an explicit non-comparability note for `context_precision` (different judge, not a regression). Historical 08-09 numbers left in place per T21's precedent |
| T31 | `ablation.py --limit` overwrites the real results file | 08-23 | `main()` now derives `out_path` with an `_smoke` suffix whenever `--limit` is set, so a smoke run lands at `ablation_<date>_smoke.json` and never shares a path with a real full run. `--rescore-modes` is unaffected — it still targets the non-smoke path. Verified live: `--limit 2 --skip-ragas` wrote `ablation_2026-08-23_smoke.json` while the real `ablation_2026-08-23.json` (hash-checked before/after) was untouched |
| T29 | `requirements-dev.txt` is unpinned and does not install cleanly | 08-23 | Pinned `ragas==0.4.3` and added `langchain-community<0.4` with a comment naming the `ragas/llms/base.py` vertexai import as the reason (confirmed live: that import only succeeds under `<0.4`, currently resolves to 0.3.31). Dev-only, never touches the Render request path. `pip install --dry-run -r requirements-dev.txt` resolves clean against the pins |
| T35 | BUG-001's `required_structure_pattern` doesn't match current answer phrasing | 08-23 | Not a phrasing issue — real evidence loss. `retriever/orchestrator.py`'s `_decompose_query()` split on *every* "and"/"vs"/"versus"/"compare" match; "What is the comparison and differences between Far Cry 5 and Assassin's Creed Valhalla" split into 3 sub-queries, the first ("What is the comparison") a boilerplate group with no real entity. `order_comparison()` (`agent/context_algorithms.py`) guarantees each `retrieval_context` group one top budget slot, so that junk group's irrelevant top chunk (Forza Horizon 5, rerank 0.016) claimed a slot ahead of the real 2nd entity — the 4000-char cap (`agent/context_assembler.py`) then pushed all 5 Assassin's Creed Valhalla chunks (rerank 0.98-0.9999) out entirely. `answer_capability=FULL` was correct (graded off the pre-assembly evidence), but the LLM never saw the Valhalla chunks and answered "no mention of Assassin's Creed Valhalla" — confirmed live before the fix. Fixed by splitting on only the LAST conjunction match, capping decomposition at exactly 2 sub-queries; verified live the assembled context now carries both entities and the answer naturally uses Gameplay/Story/World Design/Tone/Systems structure. 3 new hermetic tests (`tests/test_orchestrator.py`) |
| T32 | `--rescore-modes` can only target today's results file | 08-23 | Added `--results-file PATH` override, applied to both the read (existing-file check + merge) and the write. Defaults to the unchanged `ablation_<date>[_smoke].json` derivation when omitted, so T31's smoke-suffix behavior is untouched. Verified live: `--rescore-modes` against a deliberately nonexistent `ablation_2026-08-20.json` raises the error against that exact filename (not today's), and a `--limit`+`--results-file` smoke run wrote to the overridden path |

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
