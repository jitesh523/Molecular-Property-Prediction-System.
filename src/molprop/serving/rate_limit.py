"""
In-process token-bucket rate limiter middleware for FastAPI.

Designed for a single-worker deployment where you want a thin, dependency-
free guardrail against abuse. For multi-worker / multi-host setups, front
the API with a real rate limiter (nginx, Envoy, Redis-backed limiter).

Configurable via environment variables:

* ``MOLPROP_RATE_LIMIT`` — max requests per window (default ``120``).
* ``MOLPROP_RATE_WINDOW`` — window in seconds (default ``60``).
* ``MOLPROP_RATE_DISABLE`` — set to ``1`` to disable entirely.

Buckets are keyed by remote-IP address. Exempt paths (``/health``,
``/version``, ``/cache/stats``, ``/openapi.json``, ``/docs``, ``/redoc``)
are always allowed.
"""

from __future__ import annotations

import os
import time
from threading import RLock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

EXEMPT_PATHS = {
    "/health",
    "/version",
    "/cache/stats",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/favicon.ico",
}


class TokenBucket:
    """Classic token bucket — `capacity` tokens, refilled at `rate`/s."""

    __slots__ = ("capacity", "rate", "tokens", "last_refill")

    def __init__(self, capacity: int, rate: float) -> None:
        self.capacity = float(capacity)
        self.rate = float(rate)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def consume(self, n: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware backing the token bucket per remote IP."""

    def __init__(self, app, capacity: int = 120, window_seconds: float = 60.0):
        super().__init__(app)
        self.capacity = capacity
        self.window_seconds = window_seconds
        self.rate = capacity / window_seconds
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = RLock()
        self._disabled = os.getenv("MOLPROP_RATE_DISABLE") == "1"

    def _bucket_for(self, key: str) -> TokenBucket:
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = TokenBucket(self.capacity, self.rate)
                self._buckets[key] = b
            return b

    async def dispatch(self, request: Request, call_next):
        if self._disabled or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # Prefer X-Forwarded-For first hop, else remote IP, else "anon".
        fwd = request.headers.get("x-forwarded-for", "")
        client_key = (
            fwd.split(",")[0].strip()
            if fwd
            else (request.client.host if request.client else "anon")
        )

        bucket = self._bucket_for(client_key)
        if not bucket.consume(1):
            retry_after = int(max(1.0, 1.0 / max(self.rate, 1e-6)))
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: {self.capacity} requests "
                        f"per {int(self.window_seconds)} seconds."
                    ),
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        # Helpful headers for clients
        remaining = int(self._buckets[client_key].tokens)
        response.headers["X-RateLimit-Limit"] = str(self.capacity)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


def install_rate_limit(app) -> None:
    """Attach the rate-limit middleware to the FastAPI app from env vars."""
    if os.getenv("MOLPROP_RATE_DISABLE") == "1":
        return
    try:
        capacity = int(os.getenv("MOLPROP_RATE_LIMIT", "120"))
        window = float(os.getenv("MOLPROP_RATE_WINDOW", "60"))
    except ValueError:
        capacity, window = 120, 60.0
    app.add_middleware(RateLimitMiddleware, capacity=capacity, window_seconds=window)
