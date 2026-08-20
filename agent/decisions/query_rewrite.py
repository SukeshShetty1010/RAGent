# ============================================================
# agent/decisions/query_rewrite.py
# ============================================================
"""
agent/decisions/query_rewrite.py

Bounded, single-shot agentic decision (AUDIT_TASKS T14): given the
recent conversation, resolve the user's latest message into a
standalone query before it enters the single-turn pipeline.

This is deliberately a "condense the question" step, not history
threaded through the answer prompt: everything downstream of this
module (routing, retrieval, entity grounding, capability assessment,
prompt construction) stays exactly as single-turn as it always was.

Modeled on agent/decisions/web_search_decision.py -- the repo's
established shape for a bounded, single-shot, strict-JSON, fail-soft
LLM decision.

A deterministic pre-check runs first and skips the LLM call entirely
whenever it safely can:
  - no history at all -> passthrough, source="skipped_no_history"
  - the query is self-contained (no anaphora, not a bare fragment) ->
    passthrough, source="skipped_self_contained"

Most turns are self-contained, so the LLM only fires when a reference
actually needs resolving -- and a self-contained query can never be
corrupted by a rewriter that never runs.

Fail-soft: any failure (network error, malformed JSON, empty or
absurdly long output) falls back to the original query, tagged
source="fallback_original", so a broken rewriter degrades the system
to exactly its old single-turn behavior rather than corrupting a
retrieval.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from llm.ragent_client import chat_completion_decision

logger = logging.getLogger("QUERY_REWRITE")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# --------------------------------------------------------------
# Bounds
# --------------------------------------------------------------

# How much conversation the rewrite prompt sees. Kept in sync with the
# same bound api/main.py enforces on the wire, but re-applied here so
# this module is safe to call with an unbounded history directly (e.g.
# from a test or a future caller).
HISTORY_MAX_TURNS = 4
HISTORY_TURN_CHAR_CAP = 500

# A rewrite this long is not a query anymore -- treat it as a failure
# and fall back rather than trusting it into routing/retrieval.
MAX_REWRITTEN_QUERY_CHARS = 500

ANAPHORA_WORDS = {
    "it", "its", "that", "they", "them", "their", "this", "these",
    "those", "there",
}

# A query with fewer tokens than this is a bare fragment ("and combat?")
# that cannot be judged self-contained even without an anaphora word.
MIN_SELF_CONTAINED_TOKENS = 4


PROMPT_TEMPLATE = """You are a query-rewriting assistant for a RAG chat system.
Given the recent conversation and the user's latest message, rewrite the latest
message into a standalone query that can be understood with no conversation
history. Resolve pronouns and references (it, that, this, they, there, ...) to
the specific entity or topic being discussed. Do not answer the question --
only rewrite it so it stands alone.

Conversation history (oldest first):
{history_block}

Latest user message: {query}

Respond with STRICT JSON only, matching this schema exactly:
{{"rewritten_query": "<standalone query>", "reason": "<short justification, one sentence>"}}
"""


class QueryRewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    source: Literal[
        "llm",
        "skipped_no_history",
        "skipped_self_contained",
        "fallback_original",
    ]
    reason: Optional[str] = None


# --------------------------------------------------------------
# Deterministic pre-check
# --------------------------------------------------------------

def _needs_rewrite(query: str) -> bool:
    """True if `query` looks like it references something outside itself:
    an anaphora word, or a fragment too short to stand alone."""
    tokens = re.findall(r"\w+", query.lower())
    if any(t in ANAPHORA_WORDS for t in tokens):
        return True
    return len(tokens) < MIN_SELF_CONTAINED_TOKENS


def _format_history(history: List[Dict[str, Any]]) -> str:
    trimmed = history[-HISTORY_MAX_TURNS:]
    lines = []
    for turn in trimmed:
        role = turn.get("role", "user")
        content = (turn.get("content") or "")[:HISTORY_TURN_CHAR_CAP]
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(none)"


# --------------------------------------------------------------
# Public API
# --------------------------------------------------------------

def rewrite_query(
    *,
    query: str,
    history: Optional[List[Dict[str, Any]]],
) -> QueryRewriteResult:
    """
    Bounded, single-shot LLM decision: resolve `query` into a
    standalone form using recent conversation `history`.
    """
    if not history:
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=query,
            source="skipped_no_history",
        )

    if not _needs_rewrite(query):
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=query,
            source="skipped_self_contained",
        )

    try:
        prompt = PROMPT_TEMPLATE.format(
            history_block=_format_history(history),
            query=query,
        )

        # Deliberately no max_tokens override here -- the 320 default on
        # chat_completion_decision exists precisely because a lower cap
        # starves the Groq fallback's hidden reasoning tokens before any
        # JSON is emitted (AUDIT_TASKS T4). Re-introducing that bug here
        # would silently degrade every rewrite to fallback_original.
        raw = chat_completion_decision(
            prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        parsed = json.loads(raw)
        rewritten = str(parsed.get("rewritten_query", "")).strip()

        if not rewritten or len(rewritten) > MAX_REWRITTEN_QUERY_CHARS:
            raise ValueError(
                f"rewrite invalid: len={len(rewritten)} "
                f"(empty or exceeds {MAX_REWRITTEN_QUERY_CHARS} chars)"
            )

        return QueryRewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            source="llm",
            reason=str(parsed.get("reason", ""))[:300] or None,
        )

    except Exception as exc:
        logger.warning(f"QueryRewrite LLM call failed (fail-soft): {exc}")
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=query,
            source="fallback_original",
        )
