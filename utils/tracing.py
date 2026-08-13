# ============================================================
# utils/tracing.py
# Optional Langfuse trace layer over the engine's ProfileBlock spans.
#
# Hard no-op unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are
# both set — no import, no client, no network call. Every public
# function here fails soft: a Langfuse SDK error is logged and
# swallowed, never raised into the request path (see CLAUDE.md's
# "graceful degradation" rule).
# ============================================================
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

_client: Any = None
_client_lock = threading.Lock()
_client_init_attempted = False

# Per-thread "is a trace currently open" flag. The engine runs each
# request on its own thread (see api/main.py's run_engine thread), so
# this correctly scopes tracing to one request without touching the
# process-wide MetricsRegistry singleton.
_local = threading.local()


def _keys_present() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def get_client() -> Any:
    """Lazily construct the Langfuse client. None if keys are absent or init fails."""
    global _client, _client_init_attempted
    if _client is not None:
        return _client
    if _client_init_attempted:
        return None
    with _client_lock:
        if _client_init_attempted:
            return _client
        _client_init_attempted = True
        if not _keys_present():
            return None
        try:
            from langfuse import get_client as _langfuse_get_client

            _client = _langfuse_get_client()
        except Exception as exc:
            logger.warning(f"Langfuse client init failed (tracing disabled): {exc}")
            _client = None
    return _client


def _is_active() -> bool:
    return getattr(_local, "active", False)


@contextmanager
def trace_request(query: str):
    """Root span for one engine request. No-op context if tracing is disabled."""
    client = get_client()
    if client is None:
        yield None
        return

    _local.active = True
    _local.root = None
    cm = None
    root = None
    try:
        cm = client.start_as_current_observation(
            name="chat_request", as_type="span", input=query
        )
        root = cm.__enter__()
        _local.root = root
    except Exception as exc:
        logger.warning(f"Langfuse trace_request start failed (fail-soft): {exc}")
        cm = None
        root = None

    try:
        yield root
    finally:
        _local.active = False
        _local.root = None
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception as exc:
                logger.warning(f"Langfuse trace_request close failed (fail-soft): {exc}")


def set_trace_attributes(**kwargs: Any) -> None:
    """Attach metadata (task type, capability verdict, etc.) to the root trace span.

    No client.update_current_trace() exists in the v4 SDK; the root
    LangfuseSpan object returned by trace_request() is the update target,
    stashed in thread-local storage since callers only pass kwargs.
    """
    if not _is_active():
        return
    root = getattr(_local, "root", None)
    if root is None:
        return
    try:
        root.update(metadata=kwargs)
    except Exception as exc:
        logger.warning(f"Langfuse set_trace_attributes failed (fail-soft): {exc}")


def record_generation(
    *,
    model: str,
    prompt: str,
    output: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_ms: float,
) -> None:
    """Attach a generation observation with token usage and cost to the current trace."""
    client = get_client()
    if client is None or not _is_active():
        return
    try:
        obs = client.start_observation(
            name="LLMGeneration",
            as_type="generation",
            input=prompt,
            output=output,
            model=model,
            usage_details={"input": prompt_tokens, "output": completion_tokens},
            cost_details={"total": cost_usd},
            metadata={"latency_ms": latency_ms},
        )
        obs.end()
    except Exception as exc:
        logger.warning(f"Langfuse record_generation failed (fail-soft): {exc}")


def _on_profile_enter(name: str) -> Any:
    if not _is_active():
        return None
    client = get_client()
    if client is None:
        return None
    try:
        cm = client.start_as_current_observation(name=name, as_type="span")
        obs = cm.__enter__()
        return (cm, obs)
    except Exception as exc:
        logger.warning(f"Langfuse span enter failed for {name!r} (fail-soft): {exc}")
        return None


def _on_profile_exit(name: str, token: Any) -> None:
    if token is None:
        return
    cm, _obs = token
    try:
        cm.__exit__(None, None, None)
    except Exception as exc:
        logger.warning(f"Langfuse span exit failed for {name!r} (fail-soft): {exc}")


def register_hooks() -> None:
    """Wire this module's span hooks into utils.observability.ProfileBlock."""
    from utils.observability import set_span_hooks

    set_span_hooks(_on_profile_enter, _on_profile_exit)


def flush() -> None:
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning(f"Langfuse flush failed (fail-soft): {exc}")


# Registering here (rather than requiring every caller to remember to)
# means the hooks only exist in processes that actually import this
# module — ingestion (upsert/upsert_all.py) never imports utils.tracing,
# so its ProfileBlock("INGEST_TOTAL") spans never trigger a Langfuse
# call, orphaned trace or not. _is_active() gates it a second time
# regardless, since a trace is only "active" inside trace_request().
register_hooks()
