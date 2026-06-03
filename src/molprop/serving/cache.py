"""
Tiny in-process TTL+LRU cache for FastAPI endpoint payloads.

Designed for stateless / pure-function endpoints where the response is fully
determined by the input (e.g. /scaffold, /functional_groups, /isomers,
/standardize, /mcs, /alerts, /rgroups, /mmp).

This is intentionally dependency-free — no Redis, no aiocache — so it works
out of the box for a single-worker deployment. Multi-worker deployments
should still front the API with a CDN or Redis layer.

Usage
-----
::

    from molprop.serving.cache import cached_json

    @app.post("/scaffold")
    @cached_json(ttl_seconds=600)  # 10-minute TTL
    async def scaffold_endpoint(req: ScaffoldRequest):
        ...

The decorator inspects the function's request models (`pydantic.BaseModel`
instances) and uses their JSON dump as the cache key. Non-serialisable
arguments fall through to the underlying function (i.e. no caching).
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Callable

import logging

try:  # pydantic v2
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    BaseModel = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


class _LRUTTLCache:
    """Thread-safe LRU + TTL cache."""

    def __init__(self, maxsize: int = 512) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at <= now:
                # expired
                del self._store[key]
                self.misses += 1
                return None
            # touch (LRU)
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        with self._lock:
            self._store[key] = (time.monotonic() + ttl_seconds, value)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self._store),
                "maxsize": self.maxsize,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0


# Module-level singleton — one cache per process is enough.
endpoint_cache = _LRUTTLCache(maxsize=1024)


def _stable_key(fn_name: str, args: tuple, kwargs: dict) -> str | None:
    """Build a deterministic cache key for function arguments.
    
    Converts Pydantic models and serializable types to JSON for hashing.
    Returns None if any argument cannot be safely serialized; the call
    then bypasses the cache (graceful degradation).
    
    Args:
        fn_name: Function name for key namespace.
        args: Function arguments to serialize.
        kwargs: Function keyword arguments to serialize.
        
    Returns:
        SHA256 hex digest of serialized arguments, or None if serialization fails.
    """
    try:
        parts: list[Any] = [fn_name]
        for a in args:
            if BaseModel is not None and isinstance(a, BaseModel):
                parts.append(a.model_dump())
            elif isinstance(a, (str, int, float, bool, list, tuple, dict, type(None))):
                parts.append(a)
            else:
                return None
        for k in sorted(kwargs):
            v = kwargs[k]
            if BaseModel is not None and isinstance(v, BaseModel):
                parts.append((k, v.model_dump()))
            elif isinstance(v, (str, int, float, bool, list, tuple, dict, type(None))):
                parts.append((k, v))
            else:
                return None
        blob = json.dumps(parts, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()
    except (TypeError, ValueError) as e:
        log.debug(f"Cache key serialization failed: {e}")
        return None


def cached_json(ttl_seconds: float = 600.0) -> Callable[[Callable], Callable]:
    """Decorate an async endpoint to cache its return value for ``ttl_seconds``.
    
    Caches serializable responses based on request arguments. If arguments cannot
    be serialized or ttl_seconds is invalid, gracefully skips caching.
    
    Args:
        ttl_seconds: Cache time-to-live in seconds. Must be positive (default 600s).
        
    Returns:
        Async decorator function.
        
    Raises:
        ValueError: If ttl_seconds <= 0.
        
    Example:
        @app.post("/scaffold")
        @cached_json(ttl_seconds=600)  # 10-minute cache
        async def scaffold_endpoint(req: ScaffoldRequest):
            ...
    """
    if ttl_seconds <= 0:
        raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _stable_key(fn.__name__, args, kwargs)
            if key is not None:
                hit = endpoint_cache.get(key)
                if hit is not None:
                    return hit
            result = await fn(*args, **kwargs)
            if key is not None:
                endpoint_cache.set(key, result, ttl_seconds)
            return result

        return wrapper

    return decorator
