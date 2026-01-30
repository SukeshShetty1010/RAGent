"""
tests/caching.py

Deterministic Semantic Cache (FINAL)

Key guarantees:
- Instance methods IGNORE `self` when generating cache keys
- Explicit args / kwargs ALWAYS participate in cache keys
- Stable hashing across restarts
- Disk-backed persistence via diskcache
"""

from __future__ import annotations

import hashlib
import json
import threading
from functools import wraps
from typing import Any, Callable, Optional

try:
    from diskcache import Cache
except ImportError:
    Cache = None  # graceful fallback

from tests.observability import MetricsRegistry


# ============================================================
# Utilities
# ============================================================

def _normalize(obj: Any) -> Any:
    """
    Normalize objects into deterministic, JSON-serializable forms.
    """
    if isinstance(obj, dict):
        return {k: _normalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Fallback for non-serializable objects
    return str(obj)


def _stable_hash(*args: Any, **kwargs: Any) -> str:
    """
    Generate a stable SHA256 hash from normalized args + kwargs.
    """
    payload = {
        "args": _normalize(args),
        "kwargs": _normalize(kwargs),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ============================================================
# Semantic Cache
# ============================================================

class SemanticCache:
    """
    Thread-safe disk-backed semantic cache.
    """

    def __init__(self, path: str = ".semantic_cache") -> None:
        if Cache is None:
            raise RuntimeError(
                "diskcache is required for SemanticCache. "
                "Install with: pip install diskcache"
            )

        self._cache = Cache(path)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            self._cache.set(key, value, expire=ttl)


# ============================================================
# Cache Singleton
# ============================================================

_DEFAULT_CACHE: Optional[SemanticCache] = None
_CACHE_LOCK = threading.Lock()


def _get_cache() -> SemanticCache:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        with _CACHE_LOCK:
            if _DEFAULT_CACHE is None:
                _DEFAULT_CACHE = SemanticCache()
    return _DEFAULT_CACHE


# ============================================================
# Decorator (FIXED)
# ============================================================

def cacheable(ttl_seconds: int) -> Callable:
    """
    Decorator for deterministic caching of pure functions
    or instance methods.

    FIXES:
    - `self` is excluded from cache keys
    - Explicit args/kwargs ALWAYS affect the cache key
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            cache = _get_cache()

            # --------------------------------------------
            # FIX: Remove `self` from cache key if present
            # --------------------------------------------
            if args and hasattr(args[0], fn.__name__):
                owner = args[0].__class__.__name__
                key_args = args[1:]
            else:
                owner = "function"
                key_args = args

            key = _stable_hash(
                f"{owner}.{fn.__qualname__}",
                *key_args,
                **kwargs,
            )

            cached = cache.get(key)
            if cached is not None:
                MetricsRegistry.get().inc("cache_hits")
                return cached

            MetricsRegistry.get().inc("cache_misses")
            result = fn(*args, **kwargs)
            cache.set(key, result, ttl_seconds)
            return result

        return wrapper

    return decorator
