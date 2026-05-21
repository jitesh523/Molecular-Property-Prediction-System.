"""Tests for the in-process token-bucket rate limiter (v2.34)."""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from molprop.serving.rate_limit import RateLimitMiddleware, TokenBucket

# ── Pure token-bucket logic ──────────────────────────────────────────────────


class TestTokenBucket:
    def test_initial_capacity(self):
        b = TokenBucket(capacity=5, rate=1.0)
        for _ in range(5):
            assert b.consume(1) is True
        assert b.consume(1) is False  # depleted

    def test_refill_after_sleep(self):
        b = TokenBucket(capacity=2, rate=20.0)  # 20 tokens/sec
        assert b.consume(1) is True
        assert b.consume(1) is True
        assert b.consume(1) is False
        time.sleep(0.15)  # ≥ 2 tokens regenerate
        assert b.consume(1) is True

    def test_capacity_clamped(self):
        b = TokenBucket(capacity=3, rate=1000.0)
        time.sleep(0.05)  # large refill, but should clamp to capacity
        for _ in range(3):
            assert b.consume(1) is True
        assert b.consume(1) is False


# ── Middleware end-to-end ────────────────────────────────────────────────────


def _app(capacity: int, window: float = 60.0) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, capacity=capacity, window_seconds=window)

    @app.get("/health")
    async def health():  # exempt path
        return {"ok": True}

    @app.get("/x")
    async def x():
        return {"ok": True}

    return app


class TestMiddleware:
    def test_under_limit_passes_with_headers(self):
        client = TestClient(_app(capacity=5))
        r = client.get("/x")
        assert r.status_code == 200
        assert r.headers["X-RateLimit-Limit"] == "5"
        assert int(r.headers["X-RateLimit-Remaining"]) >= 0

    def test_over_limit_returns_429(self):
        client = TestClient(_app(capacity=3))
        for _ in range(3):
            assert client.get("/x").status_code == 200
        r = client.get("/x")
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        body = r.json()
        assert "Rate limit exceeded" in body["detail"]
        assert body["retry_after_seconds"] >= 1

    def test_health_is_exempt(self):
        client = TestClient(_app(capacity=1))
        # consume the only token
        assert client.get("/x").status_code == 200
        assert client.get("/x").status_code == 429
        # /health must still pass through
        for _ in range(5):
            assert client.get("/health").status_code == 200

    def test_x_forwarded_for_segregates_buckets(self):
        client = TestClient(_app(capacity=2))
        # Client A exhausts
        for _ in range(2):
            assert client.get("/x", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        assert client.get("/x", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429
        # Client B has its own bucket
        assert client.get("/x", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__, "-v"])
