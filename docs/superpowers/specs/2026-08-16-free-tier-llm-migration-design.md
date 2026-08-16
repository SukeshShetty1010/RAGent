# Free-Tier LLM Migration Design

**Date**: 2026-08-16
**Status**: Approved, not yet implemented

## Problem

Modal credits and Groq both exhausted. Root cause: `.github/workflows/modal-keepalive.yml`
pings the Gemma-3-12B GPU container (`gemma-3-12b-it-vllm` / `Gemma312BVLLM`) every 5
minutes, 24/7, via `scaledown_window=300`. GPU-second billing runs continuously even
though this is a rare Groq-rate-limit fallback, not the primary generator. That
keepalive burned through Modal's credits in hours.

## Constraints

- Free tier only, no budget.
- Usage volume: <100 LLM generation calls/day (dev + eval, not high-traffic prod).
- A few seconds of cold-start latency on first request after idle is acceptable —
  meaning a true serverless inference API (no container to keep warm) is strongly
  preferred over any "keep a GPU container warm" pattern, since that pattern is
  exactly what caused this problem.

## Research summary (as of 2026-08-16)

Surveyed current free-tier hosted-LLM APIs for a genuinely-free (recurring, not
trial-credit), no-idle-cost replacement:

| Provider | Free tier | Verdict |
|---|---|---|
| **Google AI Studio (Gemini)** | 15 RPM, ~1,500 req/day, no card required, never expires | **Primary** — strong instruction-following, good structured-output reliability, true serverless |
| **Groq** | 30 RPM, 14.4K req/day, no card required | **Fallback** — already integrated, keep as-is |
| Cerebras | Free tier requires payment method starting 2026-08-17; real limit only 5 RPM | Rejected |
| Together.ai | $25 one-time trial credit, not recurring | Rejected |
| OpenRouter | 20 RPM but only 50 req/day unless $10+ lifetime spend | Rejected |
| Cloudflare Workers AI | 10K Neurons/day, unpredictable per-model burn | Rejected |
| HuggingFace Inference | "Few hundred requests/hour," no published hard limit | Rejected (unreliable for a fallback path) |

Both Gemini and Groq expose an OpenAI-compatible `chat.completions.create` surface,
so Gemini slots into the existing Groq-shaped try/except pattern with minimal
code churn — only which client tries first changes.

## Architecture

- **Primary**: Gemini 2.0 Flash via Google AI Studio's OpenAI-compatible endpoint
  (`https://generativelanguage.googleapis.com/v1beta/openai/`), authenticated with
  `GEMINI_API_KEY`.
- **Fallback**: existing Groq `llama-3.1-8b-instant` client, unchanged, triggered on
  Gemini rate-limit/error.
- **Modal Gemma-3-12B fallback**: removed entirely (code, not just the keepalive).

## Files touched

| File | Change |
|---|---|
| `requirements.txt` | add `openai` (lightweight HTTP client, no heavy deps — safe for Render's 512MB limit per CLAUDE.md) |
| `llm/ragent_client.py` | new `_get_gemini_client()` (`openai.OpenAI(base_url=..., api_key=GEMINI_API_KEY)`); `chat_completion_remote` and `chat_completion_decision` try Gemini first, catch its rate-limit/error, fall to existing Groq path; remove `_get_modal_llm`, `_generate_via_modal_fallback`, `_MODAL_LLM_APP`/`_MODAL_LLM_CLASS` |
| `llm/ragent_client_streaming.py` | same primary/fallback swap for `chat_completion_streaming`; remove Modal branch |
| `llm/pricing.py` | add Gemini free-tier row (0.0 cost) so cost tracking doesn't silently drop to 0 for the new primary |
| `llm/modal_llm.py` | deleted |
| `.github/workflows/modal-keepalive.yml` | remove the Gemma-warming block only; embedder+reranker CPU warming untouched (out of scope, cheap) |
| `.env.example` | add `GEMINI_API_KEY=` |
| `tests/test_groq_client.py` | rename/adjust — Groq is now the fallback path, not primary; add equivalent live test gated on `GEMINI_API_KEY` |

## Error handling

If `GEMINI_API_KEY` is unset, `_get_gemini_client()` raises the same way
`_get_groq_client()` does today → caught → falls straight to Groq. Fail-soft,
matches CLAUDE.md's graceful-degradation rule. If both keys are unset/exhausted,
existing bare-failure behavior is unchanged — no new handling needed there.

## Out of scope

Honesty gate, quality gate, context assembly, prompt manager, and
`agent/decisions/web_search_decision.py`'s parsing logic are untouched. Gemini's
stronger instruction-following should improve structured-decision reliability,
but no code change is required there.

## Next step

Invoke `writing-plans` to turn this into a step-by-step implementation plan.
