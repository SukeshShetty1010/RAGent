# README narrative revision — working notes

Working file for the 2026-08-24 README narrative revision (architecture vs. implementation split).
**Not part of the README.** Not published, not linked from it.

**Goal:** a reader in 30 seconds can tell which parts of RAGent are the transferable pattern
(honesty gate, hybrid retrieval + rerank, entity disambiguation) and which are gaming-specific
implementation (ingestion providers, corpus, golden set) — without genericizing the project,
hiding the gaming domain, or touching a single existing metric, citation, or disclosed limitation.

**Hard rule in force for every line written here and in the README:** every file path, line
number, metric, or behavior is re-read from the actual source at the moment the sentence is
written. Nothing from memory. Anything unverifiable goes under `## Needs human input`.

---

## Section checklist

| # | README section | Lines (pre-edit) | Change | Status |
|---|---|---|---|---|
| 0 | Title + opening paragraph + badges + live demo | 1-11 | Retitle; add one bold subtitle line. Opening paragraph byte-identical. | **done** |
| 1 | The 30-second version | 15-25 | Insert split table + "why gaming" paragraph. Existing 3 paragraphs and trajectory SVG unchanged. | **done** |
| 2 | Table of contents | 29-45 | **None.** Split lives inside an existing section — no new anchors, no orphaned cross-references. | **done** (verified: no change needed) |
| 3 | Results | 49-177 | **None** — deep technical, cited, gaming-concrete by design. | **done** (verified: no change needed) |
| 4 | Architecture | 179-253 | **None.** | **done** (verified: no change needed) |
| 5 | The honesty gate, in depth | 255-279 | **None.** | **done** (verified: no change needed) |
| 6 | Data & corpus | 281-339 | Add **one** sentence on why gaming was the testbed. No numbers, diagram, or GameSpot/Cloudflare disclosure touched. | **done** |
| 7 | Golden-set review | 343-368 | **None.** | **done** (verified: no change needed) |
| 8 | Engineering decisions & what they cost | 370-414 | **None.** | **done** (verified: no change needed) |
| 9 | Reliability & fail-soft inventory | 416-429 | **None.** | **done** (verified: no change needed) |
| 10 | Observability | 431-441 | **None.** | **done** (verified: no change needed) |
| 11 | Frontend | 443-... | **None.** | **done** (verified: no change needed) |
| 12 | Quick start | ... | **None.** | **done** (verified: no change needed) |
| 13 | Deployment | ... | **None.** | **done** (verified: no change needed) |
| 14 | Testing & CI | ... | **None.** | **done** (verified: no change needed) |
| 15 | Project structure | 577-585 | **None.** | **done** (verified: no change needed) |
| 16 | Known limitations | 589-599 | Add **one** new bullet, placed first. Every existing bullet stays verbatim. | **done** |
| 17 | License | 603-605 | **None** — `LICENSE` file verified present. | **done** (verified: no change needed) |

---

## Running verification log

Appended continuously. Format: `source → what it actually says → where the README uses it`.

### Pass 1 — pre-write verification (2026-08-24)

| # | Source (verified at write time) | What it says | Used in README |
|---|---|---|---|
| V1 | `ingest/editorial_providers.py:45-49` | `PROVIDERS` tuple, exactly 3 entries, order `("gamespot", …), ("wikipedia", …), ("steam", …)` | Split table, gaming-specific column |
| V2 | `ingest/editorial_providers.py:11-14` (module docstring) | "Every provider is wrapped so a raised exception never kills the others; a game succeeds if ANY provider yields content. GameSpot stays first in the list (existing primary source, auto-reactivates once its Cloudflare block lifts); Wikipedia and Steam are additive, not replacements." | Already cited at README:286 — reused unchanged, not restated |
| V3 | `agent/capability/capability_assessor.py` | `grep -ic 'game\|gaming\|steam\|gamespot\|rawg\|igdb'` → **0** matches. Zero domain references in the whole file. | Split table, domain-agnostic column |
| V4 | `retriever/quality_gate.py:112-133` | `SOURCE_NOISE_KEYWORDS = {"sale","sales","discount","deal","bundle","price","store","buy","purchase","community","forum","thread","discussion"}`; `CONTENT_NOISE_DENSITY = 3`; `TEMPORAL_PATTERNS = [r"\b20(2[3-9]\|[3-9]\d)\b", r"\bpatch\b", r"\bupdate\b", r"\bhotfix\b", r"\bchangelog\b", r"\brelease notes\b"]` | Split table (final row) **and** the new Known-limitations bullet. **Load-bearing caveat:** these word lists are tuned for consumer-product web content. The gate's *structure* is domain-neutral; the word lists are not. Must not be claimed as portable. |
| V5 | `retriever/corpus_index.py:16-19` (module docstring) | "Token-tuple equality, not substring matching: `\"grand theft auto v\"` is a raw substring of `\"grand theft auto vi\"`, so naive `in` checks would silently pass the exact query this module exists to catch." | Split table, domain-agnostic column; "why gaming" paragraph (real collision, not hypothetical) |
| V6 | `retriever/corpus_index.py:325` | `_DEFAULT_TTL_SECONDS = 900.0`, env-overridable via `_ENTITY_INDEX_TTL_ENV` (`:341`) | Not used in the new text — verified, then dropped as unnecessary detail for a 30-second section |
| V7 | `evaluation/data/golden_set.jsonl` | **50** records; **42** with `reviewed: true`; **8** carrying a `review_note` | Split table (gaming-specific column); consistency check against README:593 (`42/50 records marked reviewed: true`) — **matches** |
| V8 | `evaluation/data/golden_set.jsonl`, `g019.review_note` | "Wrong entity: the corpus's Wikipedia-editorial match for game title \"Rust\" (Facepunch Studios, 2013) resolved to the Wikipedia article on iron oxide corrosion, not the video game. Confirmed live via Qdrant Game-collection lookup and the retrieved chunk content itself -- an identity-resolution defect (ambiguous common-word title), not a drafting error." | "Why gaming" paragraph |
| V9 | `evaluation/data/golden_set.jsonl`, `g010.review_note` | "Query \"When was Spider-Man released?\" is ambiguous and the corpus resolved it to the comic-character Wikipedia article (Amazing Fantasy #15, 1962), not a Spider-Man video game -- Marvel's Spider-Man (2018) itself is not a distinct indexed title in this corpus (only Miles Morales and Spider-Man 2 are). Entity-disambiguation gap, not a drafting error." | "Why gaming" paragraph |
| V10 | `vector/create_schema.py:65-104` | `create_collection` calls: `EditorialChunk` (hybrid dense+sparse) at `:65`, then the metadata-only loop at `:95-104` over `[…, "GameSpot_Game", "EditorialSource", "UsageCounter"]` — 7 collections total | Split table, gaming-specific column; matches README:339's existing citation |
| V11 | `LICENSE` (repo root, 1.0K) | File present. MIT badge at README:9 is backed. | Section 17 — no change needed |
| V12 | `README.md:288` | "2,563 Wikipedia chunks (91.8%) and 228 Steam chunks (8.2%) across 2,791 total, 100 games" | Subtitle corpus numbers (`100 games, 2,791 chunks`) sourced from here, not invented |
| V13 | `README.md:57-58`, `:53` | 50-query golden set, 2026-08-23, precision 1.0 / recall 1.0 / false-answer 0.0, web-on and web-off | Subtitle's "50-query golden set"; opening paragraph left byte-identical |

### Pass 2 — written claims log

**Section 0 (title + subtitle) — done.**

- Old title: `# RAGent — Capability-Aware Agentic RAG for Gaming Intelligence`
  New title: `# RAGent — Capability-Aware Agentic RAG That Refuses to Guess`
  Rationale: "for Gaming Intelligence" reads as the system's *scope*. "That Refuses to Guess" names
  the architectural property instead; the testbed is named immediately below it, so nothing is hidden.
- New subtitle, one bold line, inserted between the title and the (byte-identical) opening paragraph:
  > **The architecture — honesty gate, hybrid retrieval, entity disambiguation — is domain-agnostic by design. The implementation is gaming: built, ingested, and measured end-to-end on a 100-game / 2,791-chunk corpus and a 50-query golden set. No other domain has been tried.**
  - "domain-agnostic **by design**" — deliberately not "domain-agnostic". Design reasoning, not a
    tested result. The corpus-tuned components (V4) are why the unqualified form would be false.
  - `100-game / 2,791-chunk` ← V12 (`README.md:288`). `50-query golden set` ← V13 (`README.md:53`)
    and V7 (50 records counted in the JSONL directly).
  - "No other domain has been tried." — stated in the subtitle rather than only in Known limitations,
    so the disclaimer travels with the claim for a reader who reads nothing else.
- Opening paragraph (precision 1.0 / recall 1.0 / false-answer 0.0, 2026-08-23): **unchanged, byte-identical.**
- Badges, live-demo line: **untouched.**

**Section 1 (The 30-second version) — done.**

Placement decision: the insert goes **after** the trajectory SVG + its source line, not between the
`CapabilityAssessor` paragraph and the "That gate used to not exist…" paragraph. Reason: "That gate"
refers back to the `CapabilityAssessor` paragraph — splitting them would orphan the pronoun. The three
existing paragraphs, the SVG, and its `<sub>` source line are all **unchanged**.

Two blocks added:

1. **Split table**, 5 rows. Every cell traced to the pass-1 log:
   - Row 1 left: RRF hybrid fusion + provider-dispatched rerank — already described at README:255-279.
     Right: five providers ← V1 (`editorial_providers.py:45-49`, 3 editorial) + README:283-285 (RAWG/IGDB metadata).
   - Row 2 left: `FULL`/`PARTIAL`/`INSUFFICIENT`, "zero references to games or to any ingestion provider" ← V3
     (grep count literally 0). Right: 7 collections ← V10 (`create_schema.py:65-104`).
   - Row 3 left: token-tuple containment, not substring ← V5 (`corpus_index.py:16-19`).
     Right: 100 games / 2,791 chunks ← V12.
   - Row 4 left: provider-scoped floors + source-scoped ceiling clamp — both already documented at
     README:269-275, restated without new numbers. Right: golden-set composition **20 factual /
     10 comparison / 10 listicle / 10 unanswerable** ← counted directly from the JSONL `category`
     field this pass (`Counter({'factual': 20, 'comparison': 10, 'listicle': 10, 'unanswerable': 10})`,
     n=50, consistent with V7).
   - Row 5 is the honest one: the gate's **structure** is domain-neutral, its **word lists** are not ← V4.
     Keywords quoted are a subset of the literal set at `quality_gate.py:112-133`; none invented.
2. **"Why gaming was the testbed"** — 3 numbered reasons, then one closing design-reasoning sentence.
   - Reason 1: five providers, free structured data, fail-soft registry "exercised for real when
     GameSpot went Cloudflare-blocked" ← V2 + README:286, already-disclosed fact, not a new claim.
   - Reason 2: GTA V/VI substring ← V5; `g019` Rust/iron-oxide ← V8; `g010` Spider-Man/1962 comic ← V9.
     Both paraphrased from the `review_note` text, no detail added beyond it.
   - Reason 3: 10 unanswerable of 50 ← counted this pass (`g041`-`g050`), consistent with README:60's
     "10 true-refusals".
     **Self-caught error, corrected before the consistency pass finished.** First draft said "Ten of
     the fifty golden-set queries ask about titles that do not exist yet." Then the 10 records were
     printed: only **three** are unreleased titles (`g045` GTA VI, `g046` Elder Scrolls VI, `g047`
     Beyond Good and Evil 2). Four are games the corpus does not cover (`g041` Disco Elysium, `g042`
     Untitled Goose Game, `g043` Outer Wilds, `g044` Return of the Obra Dinn) and three are entirely
     out-of-domain (`g048` capital of France, `g049` cookie recipe, `g050` 2024 US election). Rewritten
     to name all three reasons with real examples from the file. This is the exact failure mode the
     verify-at-write-time rule exists for — the wrong version was plausible, specific, and citable.
   - Closing sentence: phrased as hypothesis. Contains the explicit sentence **"That is a hypothesis
     about the pattern, not a measurement. No non-gaming corpus has been ingested, retrieved against,
     or evaluated"** and links to `#known-limitations`. Insurance/parts/spec examples are named only as
     *failure-shape analogies*, never as domains RAGent has run on.
   - Spin-vocabulary check: no "cutting-edge / seamless / state-of-the-art / powerful / robust /
     battle-tested" in either block.

**Section 2 (Table of contents) — done, no change.** The insert lives inside the existing
`## The 30-second version` section, so no heading was added or renamed. All 15 TOC anchors and the
in-body cross-references (`#known-limitations`, `#results`, `#golden-set-review`,
`#engineering-decisions--what-they-cost`) still resolve. The new text adds one more link to
`#known-limitations`, an anchor that already existed.

**Section 6 (Data & corpus) — done.**

One sentence appended to the end of the corpus-composition paragraph (the "100% Wikipedia + Steam"
paragraph), after "…not a gap to hide.":

> Gaming was chosen as the testbed partly for that reason: five independent public providers, free and structured, are enough to build a genuinely multi-source pipeline and then find out what it does when one of them goes away — harder to arrange in a domain whose data sits behind procurement.

- "five independent public providers" ← README:283's existing "Five ingestion providers are
  implemented" + V1. No new count introduced.
- "when one of them goes away" ← the GameSpot Cloudflare block already disclosed in the same section
  (README:286) and in V2's docstring. Not a new claim.
- Untouched in this section: the 2,563 / 228 / 2,791 / 100-games numbers, the mermaid diagram, the
  GameSpot-contributes-0 disclosure, the 5-configuration table, the identity paragraph, the
  7-collections paragraph.

**Section 16 (Known limitations) — done.**

One bullet added, **first** in the list. No existing bullet removed, reordered, reworded, or softened.
Counted after the edit: **8 bullets = 1 new + 7 originals**, all 7 intact and in their original order
(golden-set review pass, KPI 4-6 fixture, `voyage` uncalibrated floors, `EXPLANATORY`
extracted-not-routed, the `106-122s` local-reranker clarification, the same-generation Results dates,
the untested `evaluation/` SDK wrappers).

New bullet's cited claims:
- `retriever/quality_gate.py:112-133` and the six quoted keywords ← V4, all literal members of
  `SOURCE_NOISE_KEYWORDS` / `TEMPORAL_PATTERNS`.
- `ingest/editorial_providers.py:45-49` ← V1.
- "No other domain has been tried" — matches the subtitle (section 0) and the 30-second version's
  closing sentence (section 1). Three statements, one fact, no drift between them.

### Pass 3 — late verification (triggered by the self-caught error above)

| # | Source (verified at write time) | What it says | Used in README |
|---|---|---|---|
| V14 | `evaluation/results/runs_2026-08-23_corpusonly.jsonl`, records `g041` / `g043` | Both report `quality_status: "quality_empty"` — the corpus-only run retrieved nothing that cleared the gate for "What's the plot of Disco Elysium?" and "What's the plot of Outer Wilds?" | Backs "games this corpus does not cover (`Disco Elysium`, `Outer Wilds`)" in the 30-second version. Written first as an inference from `should_refuse: true`; then checked against the artifact rather than left as an inference. Also cross-checked: neither title appears in any ingestion module or game list, only in evaluation query text. |
| V15 | `evaluation/data/golden_set.jsonl`, `category` field, all 50 records | `Counter({'factual': 20, 'comparison': 10, 'listicle': 10, 'unanswerable': 10})`; the 10 unanswerable are `g041`-`g050`, every one with `should_refuse: true` | Split table row 4; "why gaming" reason (3) |

---

## Consistency pass (Step 4) — completed

| Check | Result |
|---|---|
| `git diff README.md` scope | My edits are exactly **4 hunks**: `@@ -1 +1,3` (title + subtitle), `@@ -26,0 +29,14` (30-second insert), `@@ -282 +304` (Data & corpus sentence), `@@ -522 +609,2` (Known-limitations bullet). The other hunks in the diff are **pre-existing uncommitted work** from before this pass (the Golden-set-review TOC entry, the 2026-08-24 RAGAS/cost-latency re-dating, the frontend/Vitest additions) — confirmed by reading their content; none were made by this revision. No scope creep. |
| Old framing removed | `grep -c "Gaming Intelligence" README.md` → **0** |
| TOC anchors | 15 entries, all headings still present and unrenamed. No heading added. In-body links `#known-limitations`, `#results`, `#golden-set-review`, `#engineering-decisions--what-they-cost` all still resolve. The new text adds one link to `#known-limitations`, an anchor that already existed. |
| Repeated numbers identical | `2,791` ×4, `100 games` ×2 + `100-game` ×1, `50-query` ×7, `42/50` ×2, `7 Qdrant collections` ×2, `2,563` ×1, `228 Steam` ×1. Provider count reads "Five"/"five" in all three places (README:283 existing, split table, "why gaming"). No number introduced that contradicts an existing one. |
| Limitations preserved | 8 bullets = 1 new + 7 originals, originals unchanged and in original order. Nothing softened, nothing removed. |
| Markdown integrity | Fenced-block count even (no unclosed block). New table is 2 columns × 5 rows + header, pipes balanced; renders. |
| Opening paragraph | Byte-identical to pre-edit — precision 1.0 / recall 1.0 / false-answer 0.0, 2026-08-23, web-on and web-off. |
| Spin vocabulary | No "cutting-edge / revolutionary / seamlessly / state-of-the-art / powerful / robust (unqualified) / battle-tested" in any added text. |
| 30-second read test, direction A | Cannot conclude it was tested on a non-gaming domain: the subtitle says "No other domain has been tried", the 30-second close says "No non-gaming corpus has been ingested, retrieved against, or evaluated", and the first Known-limitations bullet repeats it with the corpus-tuned components named. Three statements, no drift. |
| 30-second read test, direction B | Cannot conclude it is only a gaming trivia bot: the split table is 5 rows of architecture with file citations before any gaming detail, and "why gaming was the testbed" frames the corpus as a chosen hard case, not the product. |

---

## Needs human input

Three items. None blocked the revision; each is a judgment only you can make.

1. **"No other domain has been tried" — verified against the repo, not against your history.**
   Nothing in the working tree ingests, embeds, or evaluates a non-gaming corpus: the only editorial
   providers registered are GameSpot/Wikipedia/Steam (`ingest/editorial_providers.py:45-49`), and the
   only golden set is the 50-record gaming one. If you ever ran this pipeline against other documents
   locally, in a branch, or in a scratch script that was never committed, that sentence is wrong as
   written and needs softening. Only you can answer that.

2. **The failure-shape analogy list is mine, not measured.** The 30-second close names "a near-identical
   part number, a superseded policy version, or two revisions of one spec" as the same retrieval-failure
   shape as the GTA V / GTA VI collision. That is an argument about the shape of the problem, labelled
   in the text as a hypothesis and immediately followed by the no-other-domain disclaimer. If the
   analogy overreaches for the specific companies you are approaching — or if you would rather it named
   no other industry at all — say so and it comes out; the split table stands without it.

3. **The corpus-tuned caveat is now stated twice, deliberately.** `retriever/quality_gate.py:112-133`
   appears in both the split table's last row and the new Known-limitations bullet. That repetition is
   intentional (the 30-second reader and the bottom-of-page reader should both hit it), but it is the
   single most self-critical claim in the README's first screen. If you would rather the first screen
   stay lighter, the table row can be shortened and the full version left in Known limitations — but
   the row should not be deleted outright, since it is the only thing preventing the split table from
   reading as an unqualified portability claim.
