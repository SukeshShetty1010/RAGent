# ============================================================
# llm/pricing.py
# Static USD-per-token pricing table, used to attach a cost
# estimate to each generation call without an external pricing API.
# ============================================================
from __future__ import annotations

# (usd_per_1M_input_tokens, usd_per_1M_output_tokens)
#
# Source: Groq's public pricing page (groq.com/pricing) renders
# client-side and could not be fetched as static HTML at the time this
# table was written; the numbers below are corroborated across multiple
# independent third-party pricing trackers (cloudzero.com, eesel.ai,
# aipricing.guru, getmaxim.ai) as of 2026-08-13. Verify directly against
# console.groq.com before trusting this table for a billing decision —
# it is meant for relative cost-per-query reporting in the KPI/eval
# pipeline, not for financial reconciliation.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Retired by Groq (404 as of 2026-08-17). Kept so historical eval
    # runs and traces that recorded this model still price correctly.
    "llama-3.1-8b-instant": (0.05, 0.08),
    # Current Groq fallback. $0.15/$0.60 per 1M is Groq's own rate
    # (cloudzero.com and layer3labs.io, checked 2026-08-17); the $0.75
    # output rate quoted around the web is a reseller's markup, not
    # Groq's. Same caveat as above: relative cost reporting, not
    # reconciliation. A GROQ_MODEL override with no entry here prices at
    # 0.0 rather than guessing.
    "openai/gpt-oss-120b": (0.15, 0.60),
    # Deliberate 0.0 — Gemini free tier, not a missing-model default.
    "gemini-flash-latest": (0.0, 0.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost estimate for one generation call. Unknown model -> 0.0."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    input_rate, output_rate = pricing
    return (prompt_tokens / 1_000_000) * input_rate + (completion_tokens / 1_000_000) * output_rate
