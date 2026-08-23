# RAGent — Full-Codebase Audit & Task List

**Audit date:** 2026-08-18
**Scope:** every module on the request path (`api/` → `engine/` → `agent/` → `retriever/` → `llm/` → `utils/` → `frontend/`), plus ingest, evaluation, config, Docker, and CI.
**Method:** read end to end, line by line. Claims about live behaviour were measured against the real Qdrant corpus and the real `.env`, not inferred.
**Test suite at time of audit:** `159 passed, 3 skipped` (`python -m pytest tests/`).
**Status:** all 23 tasks resolved as of 2026-08-23 (T17 last). Test suite now `273 passed, 3 skipped` full run including `live`-marked tests; `262 passed, 3 skipped` on the hermetic `-m unit` subset. Each task's "Resolved" note sits inside its own section, and the dated status updates below record what was fixed together and why.

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

**T13 and T19 fixed** together, 2026-08-20. Causally coupled — the 7 `emit_stage()` call sites in `engine/execution_engine_streaming.py` are simultaneously the forward progress signal §13 needed and the natural backward cancellation checkpoints §19 needed, so fixing them separately would have meant touching the same functions twice. See the "Resolved" notes inside §13 and §19. Test suite now `209 passed, 3 skipped` (up from 201 passed, 3 skipped/deselected — 6 net new tests in `tests/test_streaming_cancellation.py`, a file that did not exist before; with `.env` now present the two `live`-marked `test_qdrant_rebuild.py` cases that previously failed for lack of credentials now pass as well, and all 5 `live`-marked tests in the suite pass end to end).

**T7 and T12 fixed** together, 2026-08-20. Causally coupled — §7c's refusal-string divergence and §12's discarded prompt are the same `else:` branch in both engines' STEP 7, so fixing them independently meant editing that block twice and picking the constant's home twice. §7 also turned out to understate itself: it calls the blocking engine one "which nothing in production calls," but `RageEngine` is what the entire measurement apparatus (`evaluation/run_eval.py`, all five `KPI/*.py`, `tests/verify_engine.py`) runs on, so its drift from the streaming engine production actually serves made T7 a prerequisite for T17, not a peer of it. Six more divergences beyond the three originally documented (7a–c) were found during the fix and are recorded in §7's Resolved note (7d–7i), the most significant being that `tracing.set_trace_attributes(cancelled=True)` on the cancel path was a silent no-op — it ran outside the Langfuse trace's active window, so T19's cancellation attribute never reached a trace. See the "Resolved" notes inside §7 and §12 for what shipped. Test suite now `224 passed, 3 skipped` (up from 209 — 15 net new tests across `tests/test_engine_contract.py` and `tests/test_insufficient_refusal.py`, both new files).

**T18 and T14 fixed** together, 2026-08-20 (T4 folded in as a one-line freebie). Selected over the other 9 open tasks because T18 was the last unblocked Part-A defect with a live false-refusal path — T1/T3/T17/T23 are one scheduling chain blocked on the Gemini free-tier daily embedding quota, and T15/T16/T21/T22 are one-line hygiene with no behavioral stakes. Tracing §18 more precisely than it documents: `is_noise()`-dropped local chunks were never actually removed from the LLM's context (`orchestrator.run()` returns `local_chunks` unfiltered, and `execution_engine_streaming.py` assembles from `raw_chunks`) — the live harm was that a dropped chunk's `source_title` disappeared from `assess_grounding()`'s title-drift fallback, producing a false `QUALITY_EMPTY` refusal on fully-ingested games (e.g. a "Far Cry 5 combat" query, whose only titled chunk happens to mention "a great deal of freedom"). Per §1, entity grounding is currently the *only* live refusal path, so this was the entire refusal surface. `NOISE_KEYWORDS` is now split: `SOURCE_NOISE_KEYWORDS` matches only `source_title` + `source_url` (a storefront/forum is a source-shaped signal), and content only trips noise on a density rule (≥3 distinct keyword hits, not one incidental mention) — chosen to keep the existing 4-keyword storefront-blob fixture green unmodified. `MetricsRegistry.inc("chunks_dropped_as_noise")` makes the loss measurable, surfaced in the `retrieval` stage payload and the UI's pipeline detail line. T14 shipped as query condensation, not history-in-the-prompt: a new STEP 0 (`agent/decisions/query_rewrite.py`, modeled on `web_search_decision.py`) resolves anaphora into a standalone query before routing/retrieval/grounding ever run, with a deterministic pre-check that skips the LLM entirely for self-contained queries or empty history, and fails soft to the original query on any error. `history` is a per-call keyword argument on `run_streaming()`, defaulting to `None`/unaffecting every eval/KPI caller — never engine state — preserving `tests/verify_engine.py`'s statelessness contract. T4's fix (delete the `max_tokens=150` override in `web_search_decision.py`) was folded in since T14 added a sibling module reusing the same `chat_completion_decision()` default. See the "Resolved" notes inside §4, §14, and §18 for what shipped. Test suite now `239 passed, 3 skipped` (up from 224 — 15 net new tests across `tests/test_quality_gate.py`, `tests/test_query_rewrite.py` (new file), `tests/test_engine_contract.py`, `tests/test_web_search_decision.py`, and `tests/test_api.py`). The real-corpus drop-rate measurement §18's fix note calls for could not be run in this environment — no network path to the Qdrant Cloud cluster from this sandbox (connection refused/timeout on every attempt); `tests/regression_suite.py`, which also depends on a live Qdrant query, fails the same way and is unrelated to this change (confirmed: the timeout occurs inside `VectorQuery`, before any `is_noise()` code runs).

**T16 and T22 fixed** together, 2026-08-21. Selected over the other 7 open tasks because T1/T3/T17/T23 are one scheduling chain blocked on the Gemini free-tier daily embedding quota and a live Qdrant path (T3 must land before T1 can calibrate against the migrated corpus; T17/T23 both follow T1); T15 is a dead CLI-only function with zero production stakes; T21 turned out to be partly unfixable as written — `CLAUDE.md` is gitignored *and absent from disk* in this checkout, so one of its five rows has no target, and its live equivalent claims have migrated into `README.md`, which deserves its own scope decision rather than being force-fit here. T16 and T22 were done together because both edit the same block of `requirements.txt`, both change what the Docker image installs, and both are verified by the same pass — splitting them would have meant touching the dependency set twice. T16 was also treated as the higher-stakes half: it's a prerequisite for T1, since calibrating floors against a floating `fastembed` fits them to a build nothing pins.

Before touching anything, §16's risk was measured rather than assumed: the working interpreter in this environment (`py -3.10`) had **fastembed 0.8.0** installed against `hf_space/requirements.txt`'s pin of `0.7.4` — the drift §16 warns about had already happened here, undetected, because `test_hfspace_shares_local_floors` checks the floor values, not the version that produced them. A fixed query against 10 documents was scored with `TextCrossEncoder` directly under both 0.8.0 and 0.7.4 (bypassing `retriever.rag_retriever`, which never constructs the local encoder while `.env` sets `RERANKER_PROVIDER=cloudflare`): **scores were bit-identical** (max abs diff `0.0`) across all 10. The pin therefore closes a real, already-occurred drift risk rather than a hypothetical one, even though this particular version bump happened not to move the scores — a future one is not guaranteed to be as kind, and nothing before this fix could have told the difference.

§22 turned out to understate its own finding: the `500`-word default in `EditorialChunker` was never the production value — `embed/prepare_editorial_payloads.py:86` has always passed `chunk_size=300` explicitly — while `chunking/chunk_contract.md` ("~500 tokens") and `README.md` ("500 tokens, 50 overlap") both documented the unused default as if it were live *and* as if it were measured in model tokens rather than whitespace words. Real production chunks are ~300 words, roughly 150–200 model tokens — the written contract was off by about 3.3× from what ships. `LocalTokenizer`/`chunk_size`/`overlap`/`.tokenizer` were renamed to `WordSplitter`/`chunk_words`/`overlap_words`/`.splitter` throughout `chunking/editorial_chunker.py` (a pure rename — content hashing is unaffected, verified by generating chunk IDs for the same input body before and after the rename and confirming byte-for-byte match), and both docs now state the true unit and the true production value.

See the "Resolved" notes inside §16 and §22 for what shipped. Test suite now `257 passed, 3 skipped` (hermetic `-m unit` subset: `246 passed, 3 skipped` — up from 239 — 7 net new tests: `tests/test_editorial_chunker.py`, a file that did not exist before, plus two new/changed tests in `tests/test_llm_config.py`).

While verifying, a pre-existing, order-dependent failure was found in `tests/test_llm_config.py::test_reranker_model_matches_calibration` — unrelated to T16/T22 (confirmed via `git stash` that it failed identically on `main` beforehand) but fixed anyway since the root cause was cheap to isolate and fix. The test's skip check called `resolve_reranker_provider()`, a *live* read of `RERANKER_PROVIDER`, while its assertion read `retriever.rag_retriever.reranker`, an object frozen at whichever moment that module was *first imported* in the test process — deliberately, per that module's own comment, so a local reranker never gets dispatched to after having been decided against at boot. Those two sources of truth can disagree depending on which test in the session happens to trigger the first import while a `monkeypatch.setenv("RERANKER_PROVIDER", ...)` from an unrelated test is active, which is exactly what made the test fail in isolation but pass inside the full suite. Fixed by making both the skip check and the assertion read the same frozen `rag_retriever.RERANKER_PROVIDER` / `rag_retriever.reranker` state, so the test can no longer contradict itself — verified stable across isolated, `-k`-filtered, and full-suite runs.

**T15 and T21 fixed** together, 2026-08-21. Selected over the other 6 open tasks (T1, T3, T15, T17, T21, T23) because T1/T3/T17/T23 are one blocked chain — T3 needs a live Qdrant scroll plus Gemini embedding quota, T1 needs `evaluation/calibrate_relevance.py` run against the migrated corpus, T17/T23 both follow T1 — and this environment cannot execute or verify any of it: probing Qdrant Cloud with `.env` loaded, both sandboxed and with the sandbox disabled, produced `ResponseHandlingException: timed out` both times, the same condition the T18 and T13/T19 sessions recorded. T15 and T21 turned out to be genuinely coupled rather than merely both-small: T15 deletes `format_llama3_prompt()` from `retriever/rag_retriever.py`, and §21's fifth row is a stale docstring in that same file claiming two reranker providers when there are four.

`format_llama3_prompt()` — Llama-3 chat-template control tokens for a model this repo no longer runs — is gone, along with its `# Prompt Engineering (UNCHANGED)` banner. The CLI harness's `main()` now calls a new `_build_cli_prompt()` that reproduces the production engine's STEP 1/4/5/6 (`TaskRouter.route` → `RetrievalQualityGate.evaluate` → `CapabilityAssessor.assess` → `ContextAssembler.assemble` → `PromptManager.generate_prompt`) using the real collaborators, mirroring `engine/execution_engine_streaming.py:316-365` exactly — including reading `capability_reason` off `MetricsRegistry` between `assess()` and `generate_prompt()`. An `INSUFFICIENT` capability now yields a real `insufficient_prompt()` refusal instead of the harness silently routing around it. Imports of `agent.*` and `quality_gate` are local to the new function, not module-level, so `import retriever.rag_retriever` still does not pull in the `agent` package — verified by checking `sys.modules` after the import.

§21's five stale claims: `requirements.txt`'s header now names the live model defaults (`gemini-flash-lite-latest`, `openai/gpt-oss-120b`) instead of the retired ones; `vector/create_schema.py`'s `E5_VECTOR_SIZE` (a lie — the value was right, the name wasn't) is renamed `DENSE_VECTOR_SIZE`, and its docstring now says `gemini-embedding-001` instead of `E5-base-v2`; `retriever/rag_retriever.py`'s module docstring now points at `retriever/reranker_provider.py` as the single source of truth for the four providers instead of naming two of them a second time. The fifth row's original target, `CLAUDE.md`, is gitignored and absent from this checkout, so it was redirected to the live equivalent claim in `README.md` ("RAWG, IGDB, and GameSpot") — confirmed Wikipedia and Steam editorial are in fact wired into ingestion via `upsert/upsert_all.py`, `scripts/bulk_ingest.py`, and `data/wikipedia_data.py` / `data/steam_data.py` — and both are now named in the README's ETL claim, its data-ingestion diagram, and its `data/` tree comment. Left deliberately untouched: `README.md`'s API-keys line (already correct — Wikipedia/Steam need no keys) and `flagship.md` / `docs/superpowers/specs/` (historical design records, out of scope for this pass).

Two new regression tests guard three of the five claims against drifting again: `tests/test_llm_config.py::test_requirements_header_names_live_models` (parses the header, asserts it names the live `GEMINI_MODEL`/`_GROQ_MODEL` constants) and `::test_create_schema_dense_size_matches_gemini_dim` (`DENSE_VECTOR_SIZE == GEMINI_EMBED_DIM`, and `E5` no longer appears in the file). `tests/test_rag_retriever_cli.py` (new) covers `_build_cli_prompt()` end to end on stub chunks (real prompt path, no Llama-3 tokens) and its `INSUFFICIENT`/empty-evidence case, hermetically stubbing only `retriever.quality_gate._get_entity_index` — no Qdrant. See the "Resolved" notes inside §15 and §21 for the full list.

Test suite now `263 passed, 3 skipped` (hermetic `-m unit` subset: `252 passed, 3 skipped` — up from 246 — 6 net new tests). The CLI smoke test (`py -3.10 -m retriever.rag_retriever --query "..."`) cannot pass in this environment: it fails inside `RAGRetriever.retrieve()`'s `query_points` call (not at `__init__`, which succeeds without touching the network) with the same `ResponseHandlingException: timed out` as the Qdrant probe above — the wiring at that one call site is unverified end to end, though `_build_cli_prompt`'s unit tests cover everything downstream of retrieval. `py -3.10 -m tests.verify_engine` still reports `✅ ENGINE READY FOR UI` (fail-soft, degraded retrieval), unchanged from before.

**T3, T1, and T23 fixed together, 2026-08-21.** The "blocked on the Qdrant/quota chain" conclusion recorded above (and in the T15/T21 status entry before it) was itself wrong, and cost three sessions before this one caught it. `qdrant_client.QdrantClient` appends port **6333** to `QDRANT_URL` when the string carries no explicit port; this network environment blocks 6333 while allowing 443 on the exact same host. `ResponseHandlingException: timed out` was that port block, not an absent network path — appending `:443` to the local `.env`'s `QDRANT_URL` connected in 0.5s and every previously-"blocked" live check (Qdrant scroll/query, Gemini embedding, Cloudflare rerank, the full `live`-marked test subset, the CLI smoke test flagged unverified in the T15/T21 entry) now runs end to end. This is a local `.env` value fix only — every one of the ~20 call sites reads `QDRANT_URL` uniformly, and Render's own deployed `QDRANT_URL` already reaches 6333 fine, so production was never affected.

§3's prescribed fix (`--resume`) could not actually have worked even with network access: `.migration_checkpoint_gemini_embed.json` is gitignored *and absent from disk* in this checkout (same situation `CLAUDE.md` was in for T21), so `--resume` would have loaded an empty done-set and re-embedded all 2791 points instead of the 91 remaining. Built a self-detecting `--repair` mode instead (`scripts/migrate_embeddings_to_gemini.py`): E5 and Gemini vectors are both 768-dim and both L2-normalized, so neither shape nor norm tells them apart, but cosine-to-centroid does — a clean 0.65-wide empty gap separated 91 E5 outliers from 2700 Gemini inliers, exactly matching §3's recorded state. Direction (which cluster is "already migrated") is verified by re-embedding a small sample from each side and checking `self_gemini_cos ≈ 1.0`, rather than assumed from cluster size — the common failure mode once the unmigrated points are the majority. Ran live: found the predicted 91/2700 split, repaired all 91 (surviving one active 429 rate-limit wall via the existing retry/backoff), and a follow-up `--repair --dry-run` scan confirms 0 outliers remain. `tests/test_migrate_repair.py` (new, 5 hermetic tests) covers the gap-split detector and the majority-inversion case with synthetic vectors.

With the corpus uniform, ran `evaluation/calibrate_relevance.py` against `RERANKER_PROVIDER=cloudflare` (`evaluation/results/relevance_calibration_cloudflare_2026-08-21.json`, all 50 golden-set queries scored, 0 errors). Contrary to this file's own prediction that Cloudflare's scores would be "heavily saturated toward both ends," the measured distribution has real signal in the 0.3–0.9 middle (e.g. a real-but-off-topic Game identity scored 0.33) — the saturation comment in `llm/cloudflare_rerank_client.py` was written from a 2-document probe, not the calibration run. Derived `REFUSE_FLOOR=0.02` (strictly below the answerable group's minimum, 0.0249 for "Rust" — zero false refusals on the golden set, the same conservative rule `local`'s `-3.0` was derived by) and `WEAK_FLOOR=0.90` (sitting in a genuine gap in the answerable distribution, between 0.8897 and 0.9601 — lands exactly 8/40 = 20% of answerable queries `QUALITY_WEAK` and 0 `QUALITY_EMPTY`, the top of the 10–20% band targeted). Set in `retriever/quality_gate.py`'s `_FLOORS["cloudflare"]`, with the block comment above it rewritten to record the measured numbers, matching the format `local`'s entry already uses. Net effect on the 10 `should_refuse` golden queries: 7 were already caught by entity grounding (unchanged, that path runs before the floor), 1 more (`g049`, "chocolate chip cookies recipe") is now caught by `REFUSE_FLOOR`, and the remaining 2 (`g047` "Beyond Good and Evil 2" — a real Game identity with no editorial content, `g050` "2024 US presidential election") land `QUALITY_WEAK` rather than falsely-confident `QUALITY_OK` — the same accepted single-signal bound this file already documented for `g047` under the old local-scale numbers, now confirmed under Cloudflare's.

One coupled defect not listed anywhere in this file was caught and fixed in the same pass: `agent/decisions/web_search_decision.py`'s prompt hardcoded a description of the *local* cross-encoder's raw-logit scale ("<0 weak, >3 strong") into the LLM prompt that quotes `confidence_score`. That was harmless only because `QUALITY_WEAK` was unreachable before this fix, so `decide_web_search()` almost never ran. Turning the ladder on for Cloudflare means that function now runs on every `QUALITY_WEAK` query with a 0..1 score read against local's logit thresholds — a 0.33 match would misread as "<0 weak" under the old text. Added `describe_score_scale()` to `retriever/reranker_provider.py` (the existing provider single-source-of-truth module) and wired the prompt to it, so the description is correct for whichever provider is actually active. Verified live end to end: a real `decide_web_search()` call against the `g047` evidence now reports `score_scale: "normalized 0.0-1.0 — higher is better"` and returns a real LLM decision (`should_search_web=True`, `source="llm"`, not the deterministic fallback).

Turning the gate on activated the four paths this file's §1 documented as dead: `py -3.10 -m tests.verify_engine` and a live `retriever.rag_retriever` CLI run now show `Quality Gate: QUALITY_OK (..., Confidence: 1.00, ...)` with a real `max_relevance` (previously always "relevance floor skipped"); a query built from the golden set's weakest evidence (`g047`) now reaches `Capability: partial | Quality: quality_weak` and the LLM's answer includes a live `"Unsupported or Missing Parts:"` section — both previously unreachable in production. The UI's `Confidence` tile (`frontend/src/app/page.tsx:240`) required no change: it now renders a genuine 0..1 relevance score instead of a mean RRF fusion score, automatically.

§23's fix landed as part of the same pass, not separately, since it is a direct consequence: `tests/test_llm_config.py::test_cloudflare_floors_are_uncalibrated_placeholder` (asserted `_FLOORS["cloudflare"] is None`) is now `test_cloudflare_floors_are_calibrated` (asserts the floors exist, are ordered `refuse < weak`, and sit inside `0..1`). `test_voyage_floors_are_uncalibrated_placeholder` is untouched — Voyage is still genuinely uncalibrated. Added the meta-test §23 asked for: `test_active_provider_floors_are_calibrated` fails whenever `RERANKER_PROVIDER` names a provider with a `None` `_FLOORS` entry — deliberately red under `RERANKER_PROVIDER=voyage`, so "the gate is off for the active provider" can never again be an invisible state.

Test suite now `269 passed, 3 skipped` full run including all `live`-marked tests (up from the `263 passed, 3 skipped` recorded above — hermetic `-m unit` subset `258 passed, 3 skipped`, up from 252 — 6 net new tests: `tests/test_migrate_repair.py` (new file, 5 tests) plus `test_active_provider_floors_are_calibrated` in `tests/test_llm_config.py`). `py -3.10 -m tests.verify_engine` reports `✅ ENGINE READY FOR UI`. The two `test_qdrant_rebuild.py` and `regression_suite.py` failures recorded as "unrelated, no `.env`/Qdrant credentials" in earlier status entries were this same port issue — they now pass.

Remaining task: **T17** (re-run the evaluation suite) is deliberately left for a future session — it should measure the system *after* this pass, not before, and it is a separate multi-script measurement effort rather than a code fix.

## Status update — 2026-08-21 (T17, in progress)

4 of 5 steps done; the 5th is blocked on a daily API quota, not a bug. See the "In Progress" note inside §17 for the full measured numbers and the exact command to resume with.

> **Superseded by the 2026-08-23 entry below**, which finished the 5th step. Two numbers recorded here did not survive that session: the `$0.000106`/query cost is ~50× too high (a truthiness filter in `cost_latency_metrics.py` discarded every free Gemini query — see §17), and the resume instructions name `py -3.10`, an interpreter that no longer exists on this machine.

Installed `requirements-dev.txt` and found it does not install cleanly as written: latest `ragas` (0.4.3) unconditionally imports `langchain_community.chat_models.vertexai`, a module `langchain-community` removed in its 0.4.x "sunset" split into standalone packages — so the newest versions of both packages, which is what unpinned `pip install` resolves to, are mutually incompatible. Fixed by pinning `langchain-community==0.3.27` (pre-split, still ships that module) on top of the dev install; `requirements-dev.txt` itself was left unpinned since this is a dev-only, offline-eval-only dependency chain, not something worth freezing broadly for one transitive import. Full 258-test unit suite passed clean afterward.

Ran `run_eval.py` (default + `--corpus-only`, 50 golden-set queries each, live Gemini/Cloudflare/Qdrant/Tavily): 0/50 errors both runs. `refusal_metrics.py` on both: default `refusal_recall=0.7, false_answer_rate=0.3` (3/10 should-refuse queries got answered); corpus-only (no Tavily fallback) `refusal_recall=0.8, false_answer_rate=0.2` — the gap is web augmentation rescuing answers on queries the corpus-only path correctly refuses, worth a look but not investigated further this session. `cost_latency_metrics.py` on the default run: `p50=4080ms/p95=12423ms` engine latency, `$0.000106`/query mean cost (`openai/gpt-oss-120b` + Gemini pricing, correctly picked up from `llm/pricing.py`), retrieval (particularly `WebSearch`/`TavilyAPICall` and `LocalVectorSearch`) dominates the latency budget at ~80% of total.

A second real bug was found and fixed while running `ragas_eval.py --judge-backend gemini`: `evaluation/gemini_judge_llm.py`'s docstring claimed "Gemini has no n>1 restriction, so ragas's self-consistency sampling works natively here — no bypass_n needed." That claim is false for the live `gemini-flash-lite-latest` endpoint — every job failed with `OpenAIInvalidRequestError: Multiple candidates is not enabled for this model`, the identical failure mode `ragas_eval.py`'s Groq judge already has a documented workaround for (`bypass_n=True`, since "Groq's API rejects n>1"). Applied the same `bypass_n=True` fix to `build_gemini_judge()` in `evaluation/gemini_judge_llm.py` and corrected the docstring; also fixed a separate stale docstring line in `evaluation/ragas_eval.py` naming the Groq judge as `llama-3.3-70b-versatile` when the actual `JUDGE_MODEL` constant is `qwen/qwen3.6-27b`. After the fix, `ragas_eval.py` completed clean: `context_precision=0.3604, faithfulness=0.9608, answer_relevancy=0.5953` (`evaluation/results/ragas_2026-08-21_default_gemini.json`).

`ablation.py --judge-backend gemini` hit a second, different Gemini limit: not the per-minute RPM cap seen earlier, but the **daily** free-tier request cap (500 requests/day for the resolved model, `gemini-3.5-flash-lite`) — `RESOURCE_EXHAUSTED` / `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, which the earlier steps (run_eval's 50+50 queries, ragas_eval's ~30 judge calls) had already been eating into over the course of the session. `ablation.py`'s RAGAS half (`context_precision` over a 20-query × 4-mode subset, 80 judge calls) has no checkpoint, so the run was stopped rather than left retrying against an exhausted daily quota (confirmed via the tail of its output — same `429`/`RESOURCE_EXHAUSTED` on every retry, no progress). The cheap, no-LLM retrieval-only half (40 queries × 4 modes) had also not finished printing/writing by the time it was stopped, so **nothing from this run was persisted** — `evaluation/results/ablation_2026-08-21.json` does not exist yet; only the stale `ablation_2026-08-09.json` is on disk.

**To resume:** wait for Gemini's daily quota to reset (free-tier daily caps reset at midnight Pacific — roughly 19h from 2026-08-21 ~12:00 UTC, i.e. sometime around 2026-08-22 07:00 UTC), then run:

```
py -3.10 -m evaluation.ablation --judge-backend gemini
```

from the repo root with `py -3.10` (this repo's working interpreter). Everything else needed for T17 is already in place: dependencies installed and the `langchain-community` pin applied, the `bypass_n` fix already shipped so this run won't repeat the earlier failure, and `evaluation/results/runs_2026-08-21_default.jsonl` already exists for anything else that needs it. Once `ablation_2026-08-21.json` exists, T17's last measurement is done — write up the AUDIT_TASKS.md resolution note for §17 (numbers above plus the ablation result) and flip the checklist box.

## Status update — 2026-08-23 (T17 complete — all 23 tasks closed)

`ablation.py` finished: `evaluation/results/ablation_2026-08-23.json`, four modes × `n=40` retrieval / `n=20` judged, 0 dropped samples, `ragas_complete: true`. Full numbers and their comparability boundary are in §17's Resolved note; the short version is that the retrieval half improved wherever the Gemini embedding migration could reach (`dense` +0.045 precision@k) with BM25 identical to four decimals as a control, `hybrid_rerank` still does not win on precision@k, and the `context_precision` column cannot be compared to 08-09 because that run was scored by the retired Modal judge.

Reaching that result took four measurement-layer fixes, none of which touch the system under test — worth reading as a group, because the first three are the same defect at three different depths and the fourth had already corrupted a published number:

1. `ablation.py` only persisted after *both* halves finished, which is what destroyed the 08-21 attempt. It now writes after the retrieval half and after each judged mode, and stamps `ragas_complete`.
2. The Gemini judge lost samples to RPM throttling because ragas has no retry layer of its own and the OpenAI SDK's default `max_retries=2` covers ~1.5s against a 429 asking for 45s. Now 10.
3. ragas's 180s per-job timeout then cancelled samples that sat through several throttle windows, which is how `hybrid` first scored at n=16 while its neighbours scored n=20. Now `max_workers=1, timeout=900` on the Gemini path; rescored, `hybrid` returned n=20.
4. `cost_latency_metrics.py` filtered costs by truthiness, so every $0.00 Gemini query (49 of 50) was dropped and the mean was taken over the single Groq fallback. The `$0.000106`/query cost this file published on 08-21 is therefore ~50× too high; the real figure is `$0.00000212`. Fixed, artifact regenerated from the stored records, and pinned by `tests/test_cost_latency_metrics.py`.

Also added `--rescore-modes`, which repairs named modes into an existing results file (~100 judge calls instead of ~400) — the only reason the n=16 repair fit inside one day's 500-request cap.

Gemini's free tier turned out to enforce **two** independent caps, and this session hit both: 15 requests/minute (absorbed by fix 2) and 500 requests/day (a hard stop, and the same wall the 08-21 session hit). The daily window resets at midnight Pacific — 07:00 UTC — which is hours, not a day, from when it typically trips.

Test suite now `262 passed, 3 skipped` on the hermetic `-m unit` subset (up from 258 — 4 net new tests in `tests/test_cost_latency_metrics.py`, a file that did not exist before).

---

## Task checklist

Ordered by impact. Items 1–3 are the ones that change what the user actually receives.

- [x] **T1** — Calibrate the Cloudflare reranker floors; the honesty gate is currently switched off (§1) — Fixed 2026-08-21
- [x] **T2** — Order assembled context by `rerank_score`, not the RRF `score` (§2) — Fixed 2026-08-19
- [x] **T3** — Finish the last 91 chunks of the Gemini embedding migration (§3) — Fixed 2026-08-21
- [x] **T4** — Remove the `max_tokens=150` override in `decide_web_search` (§4) — Fixed 2026-08-20
- [x] **T5** — Scope `MetricsRegistry` per request, or accept cross-request KPI contamination (§5) — Fixed 2026-08-19
- [x] **T6** — Fix `last_used_model()` to report the model that served the *answer* (§6) — Fixed 2026-08-19
- [x] **T7** — Reconcile the two engines: `llm_latency_ms`, trace attributes, refusal string (§7) — Fixed 2026-08-20
- [x] **T8** — Make `candidate_spans()` handle non-interrogative queries (§8) — Fixed 2026-08-19
- [x] **T9** — Allow the entity index to refresh without a process restart (§9) — Fixed 2026-08-19
- [x] **T10** — Either consume `RouterDecision`'s three dead fields or delete them (§10) — Fixed 2026-08-19
- [x] **T11** — Decide whether web fallback should be reachable outside `TaskType.OPEN` (§11) — Fixed 2026-08-19
- [x] **T12** — Either send `insufficient_prompt()` to the LLM or delete it (§12) — Fixed 2026-08-20
- [x] **T13** — Render the `stage` SSE events in the UI (§13) — Fixed 2026-08-20
- [x] **T14** — Decide whether the product is multi-turn; wire history if so (§14) — Fixed 2026-08-20
- [x] **T15** — Delete `format_llama3_prompt()` (§15) — Fixed 2026-08-21
- [x] **T16** — Pin `fastembed` in the root `requirements.txt` (§16) — Fixed 2026-08-21
- [x] **T17** — Re-run the evaluation suite; every stored result predates the current system (§17) — Fixed 2026-08-23
- [x] **T18** — Narrow `NOISE_KEYWORDS` so ordinary review prose isn't discarded (§18) — Fixed 2026-08-20
- [x] **T19** — Cancel the engine thread when the SSE client disconnects (§19) — Fixed 2026-08-20
- [x] **T20** — Harden `_rerank()` against a short score list (§20) — Fixed 2026-08-19
- [x] **T21** — Fix stale comments and docs that describe retired infrastructure (§21) — Fixed 2026-08-21
- [x] **T22** — Drop the unused `transformers` dependency (§22) — Fixed 2026-08-21
- [x] **T23** — Invert the two "uncalibrated placeholder" tests once T1 lands (§23) — Fixed 2026-08-21

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

### Resolved — 2026-08-21

See the status update at the top of this file (T3 + T1 + T23) for the full account, including the earlier sessions' mistaken "blocked on Qdrant/quota" conclusion. Summary specific to this section: ran `evaluation/calibrate_relevance.py` against the fully-migrated corpus; the predicted saturation at the scale's extremes did not hold (real signal in 0.3–0.9), so `REFUSE_FLOOR=0.02` / `WEAK_FLOOR=0.90` were derived from the measured distribution rather than the prediction. `_FLOORS["cloudflare"]` is set in `retriever/quality_gate.py`, with the block comment above it rewritten to record the measurement. `agent/decisions/web_search_decision.py`'s prompt (a coupled defect this section didn't list) now describes the active provider's actual score scale instead of hardcoding local's raw-logit one. Test: `tests/test_llm_config.py::test_cloudflare_floors_are_calibrated` (replaces the old placeholder test, see §23).

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

### Resolved — 2026-08-21

The prescribed `--resume` fix could not have worked as written: `.migration_checkpoint_gemini_embed.json` is gitignored and was absent from disk in the checkout that fixed this, so `--resume` would have re-embedded all 2791 points, not the 91 remaining. Shipped a self-detecting `--repair` mode instead (`scripts/migrate_embeddings_to_gemini.py`) that finds the leftover E5 cluster by cosine-to-centroid gap-splitting (no checkpoint needed) and verifies direction by re-embedding a sample from each cluster rather than assuming the majority cluster is the migrated one. Also root-caused the "no network path to Qdrant" conclusion recorded in the §15/§18/§21 status entries above: `qdrant_client` defaults to port 6333, this network blocks 6333 but not 443, and appending `:443` to the local `.env`'s `QDRANT_URL` was the actual fix — see the status update at the top of this file. Ran live: found the predicted 91/2700 split, repaired all 91, confirmed 0 outliers remain on a follow-up scan. Tests: `tests/test_migrate_repair.py` (new, hermetic, 5 tests).

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

### Resolved — 2026-08-20

Deleted the `max_tokens=150` argument at the call site; the 320 default on `chat_completion_decision` now applies. Folded into the same pass as T14, since `agent/decisions/query_rewrite.py` (new, T14) calls the same function and deliberately does not pass `max_tokens` either — a regression test (`test_decide_web_search_does_not_override_max_tokens_default` in `tests/test_web_search_decision.py`, and its counterpart in `tests/test_query_rewrite.py`) asserts neither call site overrides the default, so the bug can't silently return.

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

### Resolved — 2026-08-20

Shipped the wrapper approach: `engine/execution_engine.py` is now a ~25-line `RageEngine(StreamingRageEngine)` subclass with no method overrides — `run()` was already a thin blocking wrapper around `run_streaming()` (used by `api/main.py`'s SSE loop's non-streaming callers and, before this fix, only itself), so nothing needed reimplementing. The 329-line duplicate pipeline is gone. `engine/execution_engine_streaming.py` is now the single engine body; `engine/contracts.py` (new) holds what both call sites need to agree on — `INSUFFICIENT_REFUSAL`, `GENERATION_FAILED`, `INTERNAL_ERROR` (one apostrophe, fixing 7c), the `ExecutionResult` TypedDict, and `quality_report_dict()` (replacing the two identical `_report_dict` closures).

Exploration during the fix found six more divergences beyond the three this section documented, all fixed inside `execution_engine_streaming.py`:

- **7d** — `kpis["cancelled"]` existed only on the streaming engine, so the two engines emitted different KPI *shapes*, not just different values. Now identical on both (`RageEngine` inherits the same KPI-aggregation code).
- **7e** — the blocking engine ran `validate_answer()` and `record_generation()` *inside* the generation `try`, so a validator exception silently replaced a successfully generated answer with `GENERATION_FAILED`; the streaming engine ran them outside any try, so the same exception hit the fatal handler and produced `INTERNAL_ERROR` instead. Same underlying failure, two different wrong answers, neither of which was the answer that was actually generated. Validation now has its own `try/except` (recording `output_validation_errors` on failure) so an observability bug can never overwrite a good answer.
- **7f** — the streaming engine emitted `generation`/`completed` even when the fail-soft `except Exception` branch meant `llm_ran` was `False` — the code fell through to the same `emit_stage(..., "completed")` call regardless. Now emits `"completed"` only when `llm_ran`, `"failed"` otherwise.
- **7g** — dead assignment: the `ImportError` fallback branch computed `llm_latency_ms`, then the line right after the `try/except` unconditionally overwrote it. Removed along with 7a's fix (see below) — each success path now assigns its own latency once.
- **7h** — the blocking engine had no `cancel_event`/`RequestCancelled` support, so T19's cancellation was one-sided. Free with the subclass approach: `RageEngine` inherits `run_streaming()` unchanged.
- **7i** — `tracing.set_trace_attributes(cancelled=True)` on the cancel path was a silent no-op. It ran in `except RequestCancelled`, *outside* `with tracing.trace_request(...)` — whose `finally` had already set the thread-local `active` flag to `False` by the time that line executed, and `set_trace_attributes()` returns immediately when inactive. T19's cancellation attribute never reached Langfuse. Fixed by restructuring: the pipeline body is now a `try/except RequestCancelled/finally` nested *inside* `with tracing.trace_request(...)`, and the `finally` — which always runs before the `with` block exits, on every path — writes `set_trace_attributes(llm_ran=..., output_validation=..., cancelled=...)`. This is also the fix for 7b; there is now exactly one write, on every exit path, and it always lands.

7a itself: `llm_latency_ms` is now assigned only on the two generation success paths (streaming and the `ImportError` blocking fallback), matching the blocking engine's original correct behavior; the KPI dict uses `is not None` (not truthiness) everywhere, so a genuine `0.0` survives instead of collapsing to `None`.

One consequence recorded, not hidden: `evaluation/run_eval.py` and every `KPI/*.py` script import `RageEngine` and now transitively exercise `chat_completion_streaming` (Gemini primary, Groq fallback) instead of `chat_completion_remote`. This is the point of the fix — it makes T17's eventual re-run measure the code path production actually runs, rather than a stale duplicate — but it does mean every stored evaluation result now describes a colder trail than before this change, on top of what §17 already says. No evaluation re-run was performed as part of this fix; T17 stays blocked on T3.

Tests: `tests/test_engine_contract.py` (new, 8 tests) — KPI key-set parity between the two engines (7d), `llm_latency_ms` is `None` on failure and a genuine `0.0` survives (7a), a validator exception leaves `final_answer` intact (7e), a failed generation emits `"failed"` not `"completed"` (7f), `set_trace_attributes` reaches an active fake trace on both the normal and the cancel path (7b/7i) — the assertion that fails against the pre-fix code, and both engines fall back to the identical `INSUFFICIENT_REFUSAL` object (7c). All hermetic: engines built via `object.__new__` plus stub collaborators, no network. Full suite: `224 passed, 3 skipped` (see the status update at the top of this file). `python -m tests.verify_engine` (not pytest-collected, run by hand) still passes — schema contract, silence protocol, and statelessness all confirmed against the rewritten engine.

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

### Resolved — 2026-08-20

Shipped "send it," with a verify-before-ship gate rather than sending the model's output unchecked:

- `CapabilityAssessor.assess()` (`agent/capability/capability_assessor.py`) now records a `capability_reason` categorical at each `INSUFFICIENT` return — `no_evidence`, `quality_empty:<quality.reason>`, or `comparison_entity_coverage` — so the refusal can say *why*, not just *that*.
- `insufficient_prompt(query, *, reason=None)` (`agent/prompt_templates.py`) gained the reason and an explicit "do not answer from prior knowledge, do not speculate, do not fabricate a citation" constraint. Default keeps every other caller unaffected.
- `PromptManager.generate_prompt()` gained a keyword-only `capability_reason` parameter, threaded through to `insufficient_prompt()`. The two metrics this section flagged as recorded-and-discarded (`prompt_mode`, `prompt_budget_mode`) are unchanged — they now describe a prompt that actually gets used.
- New `agent.output_validator.is_refusal(text)` — the safety gate. Accepts only non-empty, length-bounded (`MAX_REFUSAL_CHARS = 600`) text that matches no `CITATION_PATTERN`, reusing the existing citation regex rather than adding a second one. A "refusal" that cites a source is an answer that slipped past the honesty gate, not a refusal — reject it.
- `execution_engine_streaming.py`'s STEP 7 `else:` branch now calls `chat_completion_streaming(prompt, max_tokens=200, on_chunk=on_token_callback)` (so the refusal still types out live in the UI, same as a real answer), runs the result through `is_refusal()`, and on any failure — empty output, a citation slipping through, an exception, cancellation aside — falls back to the static `INSUFFICIENT_REFUSAL` constant from §7's `engine/contracts.py`. `kpis["refusal_mode"]` records which happened (`"generated"` / `"static_fallback"` / `None` on the non-refusal path), and the `generation` stage's `"skipped"` data payload carries both `capability_reason` and `refusal_mode` for the UI's existing stage-detail rendering to pick up.
- `kpis["llm_ran"]` deliberately stays `False` on this path — it means "an evidence-grounded answer was generated," and flipping it would silently change `task_success` and `answer_truncated` for every refusal in the eval set. Cost stays honest regardless: `_record_usage()` observes into the registry independent of `llm_ran`, so `kpis["cost_usd"]` still counts a generated refusal's call.

No frontend change was needed: `StageGlyph` already has a default branch for unrecognized stage status, `stageDetail` already renders `data.reason` for a `"skipped"` generation stage, and the `done` event handler already prefers `parsed.final_answer` over locally accumulated tokens — so a generated refusal that gets rejected by `is_refusal()` after partially streaming still resolves to the correct static fallback text once `done` arrives.

Tests: `tests/test_insufficient_refusal.py` (new, 7 tests) — a generated refusal that passes `is_refusal()` reaches `final_answer` with `refusal_mode == "generated"`; empty output, a citation-bearing "refusal," and a generation exception all fall back to `INSUFFICIENT_REFUSAL` with `refusal_mode == "static_fallback"`; `capability_reason` propagates into the prompt for all three `INSUFFICIENT` causes `CapabilityAssessor` can produce (using the real, unstubbed assessor); and `llm_ran`/`task_success` are `False` on every refusal path regardless of which one fired. Hermetic, no network. Fixed together with §7 — see the status update at the top of this file for why, and §7's Resolved note for the engine-collapse half both fixes share code with.

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

### Resolved — 2026-08-20

Shipped as scoped: no backend changes were needed, since the events were already correctly serialized and forwarded — only `frontend/src/app/page.tsx` changed. The single comment this section quotes is now a real branch: `handleFrame`'s `stage` case upserts into a local `stageList` (by stage name, so the `started`→`completed` pair for one stage collapses into a single row that gains a duration rather than producing two rows), mirrored into a new `activeStages` state for live rendering. A new `StageProgress` component renders it as a vertical list — status glyph, label, one-line detail pulled from `stage.data`, and a right-aligned duration (`msShort()`, a sibling to the existing `ms()` KPI helper that shows `12 ms` instead of rounding a fast stage down to `0.01 s`) — mounted next to the bot avatar while `isStreaming`, so it stays visible through the full 106–122s retrieval window this section measured, not just until the first stage arrives.

`agent_decisions` — the other half this section flagged as "computed, serialised, and dropped" — is now rendered too, via a new `AgentDecisionsPanel`: routing (task, signals, reason), retrieval strategy, the quality gate (status, confidence, evidence count, entity grounding, and `quality_pre_web` shown alongside it when a web rescue changed the verdict), the web-search decision (including `source`, so a `deterministic_fallback` — the visible symptom of open task T4 — is now visible in the UI instead of only in `MetricsRegistry`), and output validation (valid/invalid, issues, unmatched citations).

Both the stage list and the decisions panel are also kept per-message (not just live): a collapsed `<details>` after the existing `KpiPanel`, summarising `Pipeline · N stages · total duration`, so every past answer's pipeline stays inspectable in scrollback rather than disappearing once the next message arrives.

Fixed together with §19 — see the status update at the top of this file for why, and §19's Resolved note for the backend half.

Verification: `npx tsc --noEmit`, `npm run lint` (only the 10 pre-existing `@typescript-eslint/no-explicit-any` errors in `markdownComponents`, confirmed unrelated via `git stash` diff), and `npm run build` (the static export the Dockerfile's first stage depends on) all pass. No JS test harness exists in this repo, so behavior was also checked against a real Cloudflare-reranked backend with a populated `.env`: a live `/api/chat` request streamed correctly ordered `routing`→`strategy`→`retrieval` stage frames with real per-stage timings before an unrelated Qdrant-connectivity issue in the sandbox interrupted retrieval — confirming the wire format and the frontend's parsing are correct even though a full answer could not be produced in that environment.

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

### Resolved — 2026-08-20

Took the query-rewriting path, deliberately not the history-in-the-prompt path: a new STEP 0 in `execution_engine_streaming.run_streaming()` calls `agent/decisions/query_rewrite.py` (new, modeled on `web_search_decision.py`'s bounded/single-shot/fail-soft shape) to condense the latest message into a standalone query *before* routing, retrieval, entity grounding, or capability assessment ever see it. Everything downstream stays exactly as single-turn as before — no history threading through the answer prompt, no change to intent extraction or the routing cache.

A deterministic pre-check skips the LLM call entirely when history is empty or the query is already self-contained (no anaphora word, and not a bare fragment under 4 tokens), so most turns never spend the LLM budget. Any failure — exception, empty output, an absurdly long rewrite — falls back to the original query, tagged `query_rewrite_source="fallback_original"`, recorded in `MetricsRegistry` and the trace so the degradation is visible rather than silent (the lesson §4 teaches).

`history` is a `*, history: Optional[...] = None` keyword-only argument on `run_streaming()`, never engine state — `RageEngine.run()` and every eval/KPI caller keep the default and are byte-for-byte unaffected, preserving `tests/verify_engine.py`'s statelessness contract. `api/main.py`'s `ChatRequest` gained an optional `history: List[Turn] = []` field (capped at 20 turns of ≤4000 chars each server-side; the engine itself further windows to the last 4 turns of ≤500 chars when building the rewrite prompt). The frontend sends the last 4 non-failed turns from existing `messages` state and renders the rewrite (when one actually happened) in `AgentDecisionsPanel`, plus a new `query_rewrite` pipeline stage via the existing `StageProgress` panel (T13). See `tests/test_query_rewrite.py` (new), and the extended `tests/test_engine_contract.py` / `tests/test_api.py`.

One adjustment outside `query_rewrite.py` itself: adding STEP 0 made `query_rewrite` the pipeline's first checkpoint, ahead of `routing` — `tests/test_streaming_cancellation.py`'s pre-start-cancellation test now asserts the cancelled stage is `query_rewrite`, not `routing`, which is the correct new behavior (cancellation is now noticed one stage earlier), not a regression.

---

## §15 — `format_llama3_prompt()` is Modal-era leftover

**Severity:** Trivial.
**File:** `retriever/rag_retriever.py:435`

Emits Llama-3 chat-template control tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`). Nothing on the request path calls it — prompt construction lives entirely in `agent/prompt_manager.py` + `agent/prompt_templates.py`. Only `main()`, the module's CLI harness, uses it, and that harness sends the result to Gemini/Groq, neither of which uses Llama-3 control tokens. CLAUDE.md states there is no Modal dependency anywhere; this is its last residue.

**Fix:** delete it, and have the CLI harness call `PromptManager` so it exercises the real path.

### Resolved — 2026-08-21

`format_llama3_prompt()` and its `# Prompt Engineering (UNCHANGED)` banner are deleted. `main()` now calls a new `_build_cli_prompt(query, chunks)`, which reproduces the production engine's STEP 1/4/5/6 using the real collaborators — `TaskRouter().route()`, `RetrievalQualityGate().evaluate()`, `CapabilityAssessor().assess()` (reading `capability_reason` off `MetricsRegistry` immediately after, same ordering as `execution_engine_streaming.py:316-365`), `ContextAssembler().assemble()`, `PromptManager().generate_prompt()` — and returns `(prompt, task, capability, quality)` so the CLI now prints the gate's verdict instead of discarding it. An `INSUFFICIENT` capability (e.g. no evidence) now produces a real `insufficient_prompt()` refusal, per §12, rather than a special case the harness routes around.

The five `agent.*` / `quality_gate` imports live inside `_build_cli_prompt()`, not at module level, so `import retriever.rag_retriever` still does not drag in the `agent` package — verified directly: after `import retriever.rag_retriever`, `[m for m in sys.modules if m == "agent" or m.startswith("agent.")]` is empty.

`tests/test_rag_retriever_cli.py` (new) asserts `format_llama3_prompt` is gone, that no Llama-3 control token (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`) appears anywhere in the module source, that `_build_cli_prompt()` on stub chunks returns a real `PromptManager`-shaped prompt containing the query and context, and that empty chunks produce `AnswerCapability.INSUFFICIENT` plus the exact `insufficient_prompt()` text. Hermetic — the only collaborator that can reach disk on its own, `RetrievalQualityGate`'s lazy `CorpusEntityIndex`, is stubbed via `retriever.quality_gate._get_entity_index`.

Not verified end to end: `py -3.10 -m retriever.rag_retriever --query "Far Cry 5 combat"` still needs a live Qdrant. It fails inside `RAGRetriever.retrieve()`'s `query_points` call (not `__init__`, which succeeds without touching the network) with `ResponseHandlingException: timed out` — the same condition blocking T1/T3/T17/T23. The unit tests cover everything downstream of retrieval; what's unverified is the wiring at that one call site, not the logic.

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

### Resolved — 2026-08-21

`requirements.txt:12` now reads `fastembed==0.7.4`, with a comment naming why (score parity + calibrated floors) and pointing at `hf_space/requirements.txt` as its twin. `tests/test_llm_config.py::test_root_fastembed_pin_matches_hf_space` parses both files with a regex and asserts they pin the same version — not hardcoded to `0.7.4`, so a deliberate joint bump stays green without touching the test.

Parity was measured, not assumed, before the pin was applied: this environment actually had 0.8.0 installed against the 0.7.4 reference. A fixed query/10-document set scored bit-identically (`0.0` max diff) under both versions via `TextCrossEncoder` constructed directly. The floors were not, in fact, silently invalidated in this instance — but nothing before this fix could have told you that, which was the actual problem.

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

### In Progress — 2026-08-21 (superseded by the Resolved note below)

See the status update near the top of this file for the full account. `calibrate_relevance` (§1) was already done. This session ran `run_eval` (default + `--corpus-only`, 0/50 errors both), `refusal_metrics` on both (`refusal_recall` 0.7 default / 0.8 corpus-only, `false_answer_rate` 0.3 / 0.2 — web augmentation on the default run rescues some answers the corpus-only path correctly refuses), `ragas_eval` (`context_precision=0.3604, faithfulness=0.9608, answer_relevancy=0.5953`), and `cost_latency_metrics` (`p50=4080ms` engine latency, `$0.000106`/query mean cost, retrieval ~80% of total latency). Judge backend was deliberately Gemini, not Groq, per a prior-session decision (08-09's Groq run hit Groq's daily cap at 5/40 scored records).

Only `ablation.py --judge-backend gemini` remains — it hit **Gemini's own daily free-tier request cap** (500/day, distinct from the per-minute RPM limit; `GenerateRequestsPerDayPerProjectPerModel-FreeTier`), consumed in part by the four steps that already ran earlier in the same session. Its RAGAS half has no checkpoint, so the run was stopped rather than left retrying against an exhausted daily quota, and nothing from it was persisted — `evaluation/results/ablation_2026-08-21.json` does not exist yet.

A real bug was found and fixed en route: `evaluation/gemini_judge_llm.py` claimed Gemini's OpenAI-compat endpoint has "no n>1 restriction" — false; it 400s with `Multiple candidates is not enabled for this model`, the same failure Groq's judge already works around via `bypass_n=True`. Applied the identical fix to the Gemini judge and corrected both that docstring and a stale Groq-model-name docstring line in `ragas_eval.py`.

**Resume with:** `py -3.10 -m evaluation.ablation --judge-backend gemini`, after Gemini's daily quota resets (~2026-08-22, midnight Pacific). No other setup is needed — dependencies, the `langchain-community==0.3.27` pin, and the `bypass_n` fix are all already in place. Once `ablation_2026-08-21.json` exists, add its numbers to this section, flip the T17 checklist box, and update the top-of-file status section.

### Resolved — 2026-08-23

The ablation ran to completion: `evaluation/results/ablation_2026-08-23.json`, all four modes at a full `n=20` on the judged half and `n=40` on the retrieval half, `ragas_complete: true`, 0 dropped samples. That closes the last of the five re-runs this section asked for.

| Mode | Precision@K | (08-09) | Entity coverage | (08-09) | RAGAS ctx precision | (08-09) |
|---|---|---|---|---|---|---|
| `dense` | 0.9850 | 0.9400 | 0.8750 | 0.8875 | 0.4607 | 0.6537 |
| `bm25` | 0.9350 | 0.9350 | 0.8375 | 0.8375 | 0.2683 | 0.5208 |
| `hybrid` | 0.9600 | 0.9500 | 0.9125 | 0.8750 | 0.3550 | 0.6397 |
| `hybrid_rerank` | 0.9650 | 0.9200 | 0.8875 | 0.8125 | 0.3433 | 0.5168 |

Reading it, with the comparability boundary stated rather than glossed:

- **The retrieval-only half is comparable across dates** — same deterministic code, same golden set, no LLM in the loop — and it improved everywhere the embedding migration could touch. `dense` gained the most (+0.045 P@K), which is what a corpus re-embedded into Gemini's vector space (§3) should do. `bm25` is **identical to four decimals on both metrics** (0.9350 / 0.8375, both runs): the sparse path never touches the dense vectors, so this is a free control confirming the harness measured the same corpus, not a coincidence.
- **The `context_precision` column is *not* comparable across dates.** 08-09 was judged by the Modal-hosted Gemma judge (`"ragas_judge_backend": "modal"`); this run is judged by Gemini. All four modes fall by a similar 0.18–0.29, which is the signature of a stricter judge rather than a retrieval regression — a real retrieval regression would not move BM25-only and rerank-only alike while their retrieval-side metrics *improve*. Treat 08-23 as the new baseline for this metric and compare forward, not back.
- **`hybrid_rerank` still does not win on precision@k**, so `hybrid_rerank_wins_on_precision_at_k` remains `false`, as it was on 08-09. What changed is who beats it: `hybrid` (0.95) on 08-09, `dense` (0.9850) now. Reported as-is per this module's own docstring. Note the two signals disagree in the usual way — `hybrid` leads on entity coverage (0.9125) while `dense` leads on precision@k — and the production default remains `hybrid_rerank`, whose value shows up in the honesty gate's calibrated relevance floors (§1), not in top-5 precision on a golden set where nearly every mode already scores >0.93.

**Three harness defects were found and fixed while running this**, all in the measurement layer, none touching the system under test:

1. **`ablation.py` persisted nothing until both halves finished.** That is precisely why the 08-21 attempt above lost a completed retrieval half to a quota wall. It now writes the retrieval half immediately and re-writes after *each* RAGAS mode (`on_mode_done`), and records `ragas_complete` so a partial file is self-describing. This paid for itself the same day: when the daily cap hit again mid-run, 3 of 4 modes were already on disk.
2. **The Gemini judge dropped samples under RPM throttling, silently and unevenly.** ragas has no retry layer — `ragas.executor` catches the exception and records the sample as `NaN` — so the OpenAI SDK's own `max_retries` (default **2**, backing off ~1.5s total) was the only thing between a 429 asking for a 45s wait and a lost sample. Modes scored during a busy minute therefore got a smaller `n` than modes scored during a quiet one: an invisible bias in exactly the cross-mode comparison this file exists to make. `evaluation/gemini_judge_llm.py` now sets `max_retries=10` (`GEMINI_JUDGE_MAX_RETRIES` overrides).
3. **ragas's 180s per-job timeout then became the same bug one layer out.** It bounds a sample's *entire* scoring including retry waits, so a sample that sat through two or three throttle windows was cancelled anyway — observed live: `hybrid` scored `0.3344` at **n=16** while `dense` and `bm25` scored a full n=20. `RunConfig` is now `max_workers=1, timeout=900` on the Gemini path (Groq keeps `max_workers=2`). Rescored under the fix, `hybrid` came back at **n=20 → 0.3550**, confirming the short-`n` figure was a throttle artifact and that a ~0.02 difference was all it distorted.

A `--rescore-modes` flag was added for the repair itself: it re-runs the judged half for named modes only and merges into that day's existing results file, keeping the retrieval half and the other modes as measured — ~100 judge calls instead of ~400, which is what let the repair fit inside one day's quota. Its merge path was verified byte-equivalent against a backup before being used on real data.

**A fourth defect, in a number this section already published:** `cost_latency_metrics.py` built its cost list with `if r.get("cost_usd")` — a **truthiness** filter. A Gemini-served query costs exactly `0.0` (`llm/pricing.py` prices the free tier at zero deliberately, and says so), so every free query was dropped and the mean was taken over the Groq fallbacks alone. On `runs_2026-08-21_default.jsonl` that is 49 free queries discarded and **one** $0.000106 fallback kept — which is how the `$0.000106`/query figure in the In-Progress note above, and in the stored artifact, came to be ~50× the real number. Corrected to `is not None` (a never-generated hard refusal records `None`, so it stays excluded on its own merits), and `queries_priced` / `queries_at_zero_cost` were added so the provider mix behind the mean is visible rather than inferred. Recomputed from the same stored records — no re-run, no quota:

| | as published 08-21 | corrected |
|---|---|---|
| mean cost/query | $0.000106 | **$0.00000212** |
| median | $0.000106 | $0.00 |
| total for the 50-query run | $0.000106 | $0.000106 (unchanged — the sum was always right) |
| provider mix | not reported | 49/50 Gemini free tier, 1 Groq fallback |

`evaluation/results/cost_latency_2026-08-21_default.json` was regenerated in place; the pre-fix version remains in git history. `tests/test_cost_latency_metrics.py` (new, 4 tests) pins it — the two cost cases fail against the pre-fix filter and pass after, verified by stashing the fix.

**The other four steps' numbers**, re-verified from their artifacts rather than restated from the note above: `refusal_metrics` — default `precision=1.0, recall=0.7, F1=0.8235, false_answer_rate=0.3, over_refusal_rate=0.0`; corpus-only `precision=1.0, recall=0.8, F1=0.8889, false_answer_rate=0.2`. `ragas_eval` (Gemini judge, 40 answerable scored, corpus fingerprint 100 games / 2791 chunks) — `context_precision=0.3604, faithfulness=0.9608, answer_relevancy=0.5953`. `cost_latency_metrics` — engine `p50=4079.56ms / p95=12422.77ms / p99=13736.41ms`, LLM `p50=843.75ms`, with retrieval at **79.66%** of total (Tavily alone 73.12%, local vector search 47.94% — these overlap because the tree double-counts nested spans).

Two environment notes for whoever runs this next, since the resume instructions above were written for a machine that no longer matches: `py -3.10` does not exist here — the working interpreter is `RAG_env\Scripts\python.exe` (3.12), which already carries `ragas 0.4.3` + `langchain-community 0.3.31` (still pre-split 0.3.x, so the 08-21 pin's intent holds). And `QDRANT_URL` needs no `:443` on this machine; port 6333 connects in 0.86s.

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

### Resolved — 2026-08-20

Did both proposed narrowings, not one: `NOISE_KEYWORDS` is now `SOURCE_NOISE_KEYWORDS`, matched only against `source_title` + `source_url` (`is_noise(self, title, content, url="")`, positional-compatible with both existing call sites), plus a content density rule — noise only if the body contains ≥3 *distinct* keyword hits, not one incidental mention. `retriever/orchestrator.py:268`'s web call site now passes the `source_url` it already computed at line 263 but previously discarded, tightening the web path's own commerce-URL check in the same change. `MetricsRegistry.get().inc("chunks_dropped_as_noise")` runs in `evaluate()`'s filter loop and is surfaced in the `retrieval` stage's `data` payload (visible through T13's `StageProgress` panel).

Tracing the actual failure chain (more specific than this section's original text): the drop was never really invisible evidence *loss* on the local path — `orchestrator.run()` returns `local_chunks` unfiltered and `execution_engine_streaming.py` assembles context from `raw_chunks`, so a noise-dropped chunk still reached the LLM. The real, live harm was that `assess_grounding()` (`corpus_index.py`) builds its title-drift fallback only from chunks that survived `is_noise()` — so a dropped chunk's `source_title` disappeared from that check, and a query like *"Far Cry 5 combat"* against a chunk titled "Far Cry 5 Review" containing "a great deal of freedom" produced a false `QUALITY_EMPTY` → hard refusal for a fully-ingested game. Per §1, entity grounding is currently the *only* live refusal path, so this was the entire refusal surface. This regression is pinned directly: `test_noise_prose_does_not_break_entity_grounding` in `tests/test_quality_gate.py` fails against the pre-fix code and passes now.

`tests/test_quality_gate.py` gained 10 new tests: all six prose examples this section lists survive; a storefront *title* and a commerce *URL* with clean prose are still noise; the existing 4-distinct-keyword storefront-blob fixture (`test_is_noise_still_catches_real_noise_keywords`) stays green **unmodified** — the density threshold of 3 was chosen to keep it green, not the other way around; and the metrics counter increments correctly.

Not done: the real-corpus drop-rate measurement (old-rule vs. new-rule, by keyword) this section's own audit standard calls for ("measured against the real corpus, not inferred"). This sandbox has no network path to the Qdrant Cloud cluster the corpus lives in — every attempt (`tests.regression_suite`, `tests.verify_engine`, a scratch `client.scroll()` script) times out or is refused at the TCP level, independent of this change. `tests.verify_engine` still reports `✅ ENGINE READY FOR UI` (fail-soft as designed), and the full hermetic suite is green, but the corpus-level number remains outstanding until this runs somewhere with Qdrant access.

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

### Resolved — 2026-08-20

Shipped exactly the two-part fix this section recommended. `api/main.py`'s `event_generator` no longer calls `q.get()` unbounded; it polls with `q.get(timeout=1.0)`, catching `queue.Empty` and re-checking `request.is_disconnected()` each iteration, so a thread-pool worker is now parked for at most 1s instead of the full 106–122s retrieval. A `threading.Event` (`cancel`) is created per request and set in a `finally` around the generator, covering both the explicit disconnect break and the `GeneratorExit`/`CancelledError` sse_starlette raises when it notices the disconnect first.

The cancellation flag itself is `cancel_event`, a new optional keyword argument on `StreamingRageEngine.run_streaming()` (defaulted to `None`, so `run()` and every CLI/KPI caller is unaffected). A `checkpoint(next_stage)` closure next to the existing `emit_stage` raises a new `RequestCancelled(next_stage)` when the event is set, and is called immediately before each of the 7 `emit_stage(..., "started")` sites — so a stage that never runs also never emits a spurious `started` event. Two traps flagged during planning were handled explicitly rather than left to the existing exception handling: an `except RequestCancelled: raise` guard was added ahead of STEP 7's inner `except Exception`, which otherwise would have reported "I could not generate a response at this time" for a cancellation; and a dedicated `except RequestCancelled as exc` handler was added ahead of the outer fatal-error handler, which otherwise would have logged "Fatal execution error" and set the internal-error answer. The token loop also checks the flag per chunk, so a disconnect mid-answer stops pumping tokens into a queue nobody drains — breaking out of `chat_completion_streaming` mid-iteration raises `GeneratorExit` inside it, a `BaseException` that passes cleanly through that function's `except Exception`, so no unwanted Groq fallback is triggered by a cancellation.

Per this section's own scope note, mid-stage cancellation was deliberately not attempted — retrieval itself stays uninterruptible; what this buys is skipping everything *after* the stage boundary where the disconnect is next noticed. `kpis["cancelled"]` was added (`False` by default) so the KPI payload records whether a given run ended in cancellation rather than success or a genuine error.

Fixed together with §13 — see the status update at the top of this file for why, and §13's Resolved note for the frontend half that consumes the same stage boundaries.

Tests: `tests/test_streaming_cancellation.py` (new, 6 tests) — cancellation already signalled before `run_streaming` starts (nothing beyond the routing checkpoint runs), cancellation noticed mid-retrieval via a stub `orchestrator.run` that sets the event itself (the actual §19 scenario — retrieval completes, but nothing after the next checkpoint does), the default `cancel_event=None` path still runs the full pipeline, a genuine exception is still classified fatal and not cancelled (proves the exception ordering), and two API-level tests against a fake engine confirming `event: stage` frames precede `event: done` and that a stage arriving only after the 1s poll interval elapses is neither dropped nor causes the loop to spin. Full suite: `209 passed, 3 skipped` (see the status update at the top of this file).

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

### Resolved — 2026-08-21

All five rows fixed, with the fourth redirected: `CLAUDE.md` is gitignored *and absent from disk* in this checkout, so its row had no target. Its live equivalent — the same "RAWG, IGDB, GameSpot" ingestion claim — lives in `README.md`, so that became the fix target instead.

- `requirements.txt`'s header now reads `gemini-flash-lite-latest` (primary) / `openai/gpt-oss-120b` (fallback) — the live defaults in `llm/gemini_client.py:35` / `llm/ragent_client.py:30` — instead of the two retired models. `tests/test_llm_config.py::test_requirements_header_names_live_models` parses the header and asserts it names both live constants, so a future model swap that forgets the header turns the build red.
- `vector/create_schema.py`'s `E5_VECTOR_SIZE` is renamed `DENSE_VECTOR_SIZE` (both the constant and its one use site); the docstring's `"dense": E5-base-v2 (768-dim, cosine)` now reads `gemini-embedding-001`. `tests/test_llm_config.py::test_create_schema_dense_size_matches_gemini_dim` asserts `DENSE_VECTOR_SIZE == GEMINI_EMBED_DIM` and that `E5` no longer appears anywhere in the file.
- `retriever/rag_retriever.py`'s module docstring no longer names two providers; it points at `retriever/reranker_provider.py`, which already documents all four (`local`, `hfspace`, `cloudflare`, `voyage`) and their score scales in one place, rather than becoming a third copy of the same list.
- The `CLAUDE.md` row, redirected: `README.md`'s "Multi-source ETL from RAWG, IGDB, and GameSpot APIs" claim, its Data Ingestion mermaid diagram, and its `data/` tree comment now also name Wikipedia and Steam — confirmed wired into ingestion via `upsert/upsert_all.py`, `scripts/bulk_ingest.py`, and the presence of `data/wikipedia_data.py` / `data/steam_data.py`. `README.md`'s separate API-keys line ("RAWG, IGDB, GameSpot, Tavily") was left as-is — it's already correct, since Wikipedia and Steam need no keys.

Left out of scope, stated rather than silently skipped: `flagship.md` and `docs/superpowers/specs/` carry their own dated claims (Modal verification steps, "~500 tok" chunking); they're historical design records, not live documentation, and were judged not worth touching in this pass.

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

### Resolved — 2026-08-21

`transformers` deleted from `requirements.txt` (kept the adjacent sentence-transformers/torch-absence note — still true, still load-bearing). Decided against the tokens option: `chunking/editorial_chunker.py`'s `LocalTokenizer`/`encode()`/`chunk_size`/`overlap`/`self.tokenizer` are now `WordSplitter`/`split()`/`chunk_words`/`overlap_words`/`self.splitter`, and the docstring states plainly that it returns words, not model tokens. The `500` default was left as-is (not changed to `300`) but now carries a comment that the only caller passes `300` explicitly, so the default no longer misdescribes production.

This section understated its own finding: `embed/prepare_editorial_payloads.py:86` has always passed `chunk_size=300`, not the `500` default the docs describe. `chunking/chunk_contract.md` and `README.md:204` both said "500 tokens" — wrong on the unit (words, not tokens) and wrong on the value (300, not 500) at once. Both now say "300 words, 50 overlap" with the ~150–200-token approximation noted in `chunk_contract.md`.

This is a pure rename — `_deterministic_uuid` hashes chunk content, not parameter names, and no chunk boundary moved. Verified directly: chunk IDs generated for an identical input body from the pre-rename code (`git show HEAD:chunking/editorial_chunker.py`) and the post-rename code matched byte-for-byte across all 4 chunks of a 1000-word test body. No re-chunk or re-embed was needed.

New test file `tests/test_editorial_chunker.py` (none existed for this module before) covers `WordSplitter.split()` word counting, chunk count/stride at a small window, that `chunk_words` is honored as a word count rather than a char/token count, the `overlap_words >= chunk_words` assert, and that the production configuration (`chunk_words=300, overlap_words=50`) constructs and chunks correctly.

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

### Resolved — 2026-08-21

Landed together with §1 (see the status update at the top of this file), not separately — this section's fix is a direct, necessary consequence of §1 shipping. `test_cloudflare_floors_are_uncalibrated_placeholder` is now `test_cloudflare_floors_are_calibrated`: asserts the floors exist, are ordered `refuse < weak`, and sit inside `0..1`. `test_voyage_floors_are_uncalibrated_placeholder` is untouched — Voyage is still genuinely uncalibrated. Added the suggested meta-test as `test_active_provider_floors_are_calibrated`: reads `resolve_reranker_provider()` live and fails if that provider's `_FLOORS` entry is `None` — deliberately red under `RERANKER_PROVIDER=voyage`.

---

# Summary

**Nothing is crashing.** Every finding here is code that runs, returns, and reports success while not doing its job.

The three that change what users receive:

1. **§1 — the honesty gate is off.** *(Fixed 2026-08-21, together with §3 and §23 — see status update at the top of this file.)* An uncalibrated `None` disabled the relevance ladder for the active reranker, so every query with any evidence was graded `QUALITY_OK`, PARTIAL was nearly unreachable, and the weak-evidence web-search path was dead. `_FLOORS["cloudflare"] = (0.02, 0.90)`, measured against the fully-migrated corpus.
2. **§2 — reranking is thrown away.** *(Fixed 2026-08-19, together with §20 — see status update at the top of this file.)* Context assembly re-sorts by the RRF score, then a 4000-char budget admits 2–3 of the ~1500-char chunks. The cross-encoder pays full cost and barely influences the prompt.
3. **§3 — the migration is 91 chunks short.** *(Fixed 2026-08-21 — see status update at the top of this file.)* Those chunks were being searched with mismatched embedding spaces; a self-detecting `--repair` mode found and re-embedded all 91, confirmed by a follow-up 0-outlier scan.

**Order of operations mattered:** §3 → §1 → §17. Calibrating or evaluating before the corpus is uniform bakes the mismatch into the floors and into every published number. §3 and §1 are done, in that order; §17 (re-running the evaluation suite against the now-fixed system) is the remaining step.

The recurring theme in Part B is a pipeline that computes more than it consumes — routing fields nobody reads, prompts built and discarded, stage events streamed and ignored, an agentic decision that cannot be reached. Each is individually small. Together they mean the system's observable behaviour is substantially simpler than its architecture implies, and the difference is not visible from the outside.
