# Reranker Latency & Embedder Quota — Free-Tier Options

**Date**: 2026-08-16
**Status**: Undecided — options gathered, no option chosen. Resume here next session.

## Problem

Two separate pain points, both stem from Render's free tier (0.1 vCPU / 512MB) and the
post-Modal architecture:

1. **Reranker latency/OOM**: in-process `fastembed` cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`)
   in `retriever/rag_retriever.py` is CPU-throttled on Render — real query retrieval+rerank
   measured ~106-122s live vs ~1-3s locally. `_RERANK_BATCH_SIZE = 1` is already a load-bearing
   hack to avoid OOM (see CLAUDE.md Environment rules).
2. **Gemini embedding migration quota wall**: `scripts/migrate_embeddings_to_gemini.py` is
   re-embedding stored `EditorialChunk` vectors into Gemini's space, blocked by Gemini's free-tier
   ~1,500 req/day cap. Resumable (`--resume`), currently at 800/2791 per last check-in. Not
   actually a blocker long-term — it just takes several days of `--resume` runs across daily
   quota resets.

Source plan that kicked this off: `c:\Users\sukes\.gemini\antigravity\brain\f3afa23d-bd32-48bf-b932-3781076ec998\implementation_plan.md`
(proposed Voyage AI managed API vs. Hugging Face Spaces self-hosting).

## Constraints

- Must stay 100% free (explicit user requirement — same constraint that drove the
  Modal→Gemini/Groq migration, see `2026-08-16-free-tier-llm-migration-design.md`).
- Don't reintroduce the class of problem just removed: an external host that needs
  keep-warm babysitting (that's what killed Modal — GPU-second billing from a keepalive cron).
- Preserve fail-soft behavior — a slow/unreachable third-party call must degrade
  (skip rerank → fall back to RRF `score`), never crash or hang the SSE stream.
- Swapping the reranker model invalidates `evaluation/results/relevance_calibration_2026-08-12.json`
  — must re-run calibration against `retriever/quality_gate.py`'s `REFUSE_FLOOR`/`WEAK_FLOOR`
  before shipping any reranker swap (CLAUDE.md rule).
- Swapping the embedder changes Qdrant vector dimension — `EditorialChunk` collection in
  `vector/create_schema.py` needs recreation, not just re-population, and abandons the
  in-flight Gemini migration progress (800/2791 done).

## Options considered

### Option A — Voyage AI, reranker only (embeddings stay on Gemini)
Swap just the reranker to Voyage's `rerank-2.5-lite` API. Keep Gemini embeddings and let
`migrate_embeddings_to_gemini --resume` finish over several days as designed.

- Pros: smallest blast radius, no vector-dim change, no abandoned migration work, kills the
  Render CPU/OOM problem outright (sub-second HTTP call, no in-process ONNX model).
- Cons: reranker model swap still requires calibration re-run. New third-party dependency
  (Voyage API key) but with a clean fail-soft path (unreachable → fall back to RRF `score`,
  `quality_gate.py` already keys off that field).
- **This was the standing recommendation before the user asked to reconsider full Voyage.**

### Option B — Voyage AI, embeddings + reranker (full swap)
Move both embedding and reranking to Voyage. Free tier: 50M tok/mo embeddings, 200M tok/mo
reranking; corpus is ~1.5M tokens, migrates in seconds.

- Pros: single managed API, always-warm (no cold-start / keep-alive problem at all — better
  than HF Spaces on this axis), removes fastembed/ONNX from the Render process entirely
  (RAM headroom back, could allow raising reranked-candidate count, currently capped by CPU
  throttle per CLAUDE.md).
- Cons:
  - Vector dim changes (Gemini 768-dim → Voyage's dim, e.g. voyage-3-lite=512 /
    voyage-3=1024) — `vector/create_schema.py` `EditorialChunk` collection must be recreated.
  - Abandons in-flight Gemini migration (800/2791) — a third embed pass on the same corpus.
  - Touches `llm/gemini_client.py` embed callers: `chunking/editorial_chunker.py`,
    `embed/prepare_editorial_payloads.py`, `retriever/rag_retriever.py` (query-time embed),
    and requires rewriting `scripts/migrate_embeddings_to_gemini.py` for Voyage.
  - No clean fail-soft for the embedder specifically: unlike the reranker (degrade to RRF
    `score`), a fallback embedder would live in a different vector space than the corpus —
    query/doc mismatch breaks retrieval outright rather than degrading gracefully.
  - New single point of failure with no fallback path for embeddings.

### Option C — Hugging Face Spaces (self-hosted, free CPU Basic)
Host the existing fastembed embedder + cross-encoder reranker on a free HF Space
(2 vCPU / 16GB RAM), call it via HTTP from Render.

- Pros: keeps current models (no re-embed, no calibration re-run), $0, no card required.
- Cons: reintroduces an external host to babysit — exactly the pattern that caused the
  Modal blowup (keepalive cron burning through free-tier limits). Free Spaces sleep after
  48h idle; cold wake is a container reboot + model reload (tens of seconds, not instant).
- **Cold/warm mitigation** (cheap, reuses existing infra):
  1. Extend `.github/workflows/render-keepalive.yml`'s existing 10-min cron (already pings
     Render's `/ping`) to also ping the HF Space's health endpoint — well under the 48h
     sleep threshold, so it never actually sleeps. Zero new infra, zero new cost.
  2. Set an explicit HTTP timeout (~8-10s) + one retry on Render's call to the Space, so an
     Actions-runner outage or missed ping can't hang the SSE stream indefinitely.
  3. Fail-soft on timeout: reranker → skip rerank, fall back to RRF `score` (clean). Embedder
     → no clean fallback (same vector-space mismatch issue as Option B) — surface a degraded
     result for that request rather than crash.
  - With #1 in place, cold-start is effectively a non-issue in practice; #2/#3 are just
    belt-and-suspenders for the rare case the ping itself fails.

## Cost / payment check

- **HF Spaces (CPU Basic)**: confirmed $0, no card, no billing trigger unless hardware/storage
  is manually upgraded.
- **Voyage AI**: free tier (50M tok/mo embed, 200M tok/mo rerank) is *recurring*, not one-time —
  live queries burn quota too (1 query-embed + 1 rerank call per chat query), so low-traffic
  usage is nowhere near the cap, but this isn't purely a one-time migration cost.
  **Unverified**: whether Voyage requires a card on file for the free tier, and whether it
  hard-stops or auto-bills past the free cap. Must check on Voyage's signup page before
  committing — not confirmable from here (pricing/signup terms drift over time).
- **GitHub Actions cron pings**: already budgeted (existing Render keepalive does this);
  adding an HF Space ping to the same workflow is free-tier-negligible.
- **Qdrant**: no new cost regardless of option (same collection, existing plan).

## Open questions for next session

1. Check Voyage AI's actual signup flow — card required? auto-bill or hard-stop past free tier?
2. Pick: Option A (reranker-only, low risk) vs Option B (full swap, more upside but abandons
   in-flight Gemini migration + no embedder fallback) vs Option C (HF Spaces, keeps current
   models but reintroduces a host to manage, mitigated by keepalive extension).
3. If Option B or C chosen: plan the `vector/create_schema.py` / Qdrant collection recreation
   and re-embed sequencing (can't run old and new vector spaces side by side in one collection).
4. Either way (A, B, or C touching the reranker): schedule a calibration re-run against
   `retriever/quality_gate.py`'s `REFUSE_FLOOR`/`WEAK_FLOOR` before shipping.

## Recommendation as of this session

Leaning **Option A** (Voyage reranker-only) as the safe default — smallest change, clean
fail-soft, solves the actual Render CPU/OOM pain without abandoning the Gemini embedding
migration already in progress. Option B is attractive mainly because Voyage's always-warm
API removes the cold-start problem Option C needs mitigation for — but costs a second full
corpus re-embed and loses the embedder fail-soft path. User has not decided; revisit fresh
next session with the Voyage signup question answered.
