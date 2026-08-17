"""
utils/usage_counter.py

Thread-safe, fail-soft request counters for Gemini/Groq usage, shared
across Render chat traffic and local RAGAS/eval runs via a small Qdrant
collection ("UsageCounter"). Modeled on utils/observability.py's
MetricsRegistry, but flushed to Qdrant instead of staying purely
in-process, so the two environments aggregate into one dashboard.

This is telemetry, not a control path — every Qdrant call here is
wrapped so a counter failure can never break a query or an eval run.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

COLLECTION = "UsageCounter"
PROVIDERS = ("gemini", "groq", "voyage", "hfspace", "cloudflare")
SURFACES = ("chat", "decision", "embedding", "ragas_judge", "ablation", "rerank")

# Configured from the actual AI Studio / Groq console quota pages, not a
# blog post — verify before trusting these for anything but a rough
# "how close am I" dashboard bar. Voyage is deliberately absent: its free
# allowance is a token budget, not a requests-per-day cap, so an "rpd"
# bar would be meaningless. read_today() tolerates the omission — the
# provider renders with limits {} and percent_of_rpd None.
FREE_TIER_LIMITS = {
    "gemini": {"rpd": 1500, "rpm": 15},
    "groq": {"rpd": 14400, "rpm": 30},
}

_NAMESPACE = uuid.UUID("6f6b1e2a-6c1a-4b0a-9c3a-6b1e2a6c1a4b")

Key = Tuple[str, str, str]  # (provider, surface, utc_date)


def _point_id(key: Key) -> str:
    provider, surface, date = key
    return str(uuid.uuid5(_NAMESPACE, f"{provider}:{surface}:{date}"))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class UsageCounter:
    _instance: "UsageCounter | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._counts: Dict[Key, Dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "fallback_events": 0}
        )
        self._registry_lock = threading.Lock()

    @classmethod
    def get(cls) -> "UsageCounter":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # --------------------------------------------------------
    # Recording (in-process, zero I/O — safe on the hot path)
    # --------------------------------------------------------
    def record(
        self,
        provider: str,
        surface: str,
        *,
        count: int = 1,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        fallback: bool = False,
    ) -> None:
        key = (provider, surface, _today())
        with self._registry_lock:
            entry = self._counts[key]
            entry["requests"] += count
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
            if fallback:
                entry["fallback_events"] += count

    # --------------------------------------------------------
    # Flush — read-modify-write into Qdrant, per dirty key
    # --------------------------------------------------------
    def flush(self) -> bool:
        """Push accumulated in-process counts into Qdrant and clear them.
        Returns True on success, False on any failure (never raises)."""
        with self._registry_lock:
            pending = dict(self._counts)
            self._counts.clear()

        if not pending:
            return True

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import PointStruct

            client = QdrantClient(
                url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
                api_key=os.environ.get("QDRANT_API_KEY") or None,
                timeout=10,
            )
            try:
                for key, delta in pending.items():
                    point_id = _point_id(key)
                    existing = client.retrieve(
                        collection_name=COLLECTION, ids=[point_id], with_payload=True
                    )
                    payload = dict(existing[0].payload) if existing else {
                        "provider": key[0],
                        "surface": key[1],
                        "date": key[2],
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "fallback_events": 0,
                    }
                    for field in ("requests", "prompt_tokens", "completion_tokens", "fallback_events"):
                        payload[field] = payload.get(field, 0) + delta[field]

                    client.upsert(
                        collection_name=COLLECTION,
                        points=[PointStruct(id=point_id, vector=[0.0], payload=payload)],
                    )
                return True
            finally:
                client.close()
        except Exception as exc:
            logger.warning(f"UsageCounter flush failed (fail-soft, counts kept for retry): {exc}")
            # Put the pending deltas back so a later flush can retry them.
            with self._registry_lock:
                for key, delta in pending.items():
                    entry = self._counts[key]
                    for field in delta:
                        entry[field] += delta[field]
            return False

    # --------------------------------------------------------
    # Reading — for the /api/usage dashboard endpoint
    # --------------------------------------------------------
    def read_today(self) -> Dict[str, object]:
        """Today's totals per provider/surface, read from Qdrant (source
        of truth across both Render and local RAGAS runs). Fail-soft: on
        any Qdrant error, returns in-process-only numbers with a
        "degraded" flag rather than raising."""
        today = _today()
        degraded = False
        by_provider: Dict[str, Dict[str, int]] = {
            p: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "fallback_events": 0}
            for p in PROVIDERS
        }
        by_surface: Dict[str, Dict[str, int]] = {}

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue

            client = QdrantClient(
                url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
                api_key=os.environ.get("QDRANT_API_KEY") or None,
                timeout=10,
            )
            try:
                records, _ = client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="date", match=MatchValue(value=today))]
                    ),
                    limit=len(PROVIDERS) * len(SURFACES) + 10,
                    with_payload=True,
                    with_vectors=False,
                )
                for r in records:
                    payload = r.payload or {}
                    provider = payload.get("provider")
                    surface = payload.get("surface")
                    if provider not in by_provider:
                        continue
                    for field in ("requests", "prompt_tokens", "completion_tokens", "fallback_events"):
                        by_provider[provider][field] += payload.get(field, 0)
                    surface_entry = by_surface.setdefault(
                        surface, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
                    )
                    surface_entry["requests"] += payload.get("requests", 0)
                    surface_entry["prompt_tokens"] += payload.get("prompt_tokens", 0)
                    surface_entry["completion_tokens"] += payload.get("completion_tokens", 0)
            finally:
                client.close()
        except Exception as exc:
            logger.warning(f"UsageCounter read_today failed (fail-soft): {exc}")
            degraded = True
            # Fold in-process (unflushed) counts so the dashboard still
            # shows something even when Qdrant is unreachable.
            with self._registry_lock:
                for (provider, surface, date), delta in self._counts.items():
                    if date != today or provider not in by_provider:
                        continue
                    for field in ("requests", "prompt_tokens", "completion_tokens", "fallback_events"):
                        by_provider[provider][field] += delta[field]

        return {
            "date": today,
            "degraded": degraded,
            "by_provider": {
                provider: {
                    **counts,
                    "limits": FREE_TIER_LIMITS.get(provider, {}),
                    "percent_of_rpd": (
                        round(100 * counts["requests"] / FREE_TIER_LIMITS[provider]["rpd"], 1)
                        if FREE_TIER_LIMITS.get(provider, {}).get("rpd")
                        else None
                    ),
                }
                for provider, counts in by_provider.items()
            },
            "by_surface": by_surface,
        }
