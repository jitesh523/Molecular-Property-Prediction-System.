"""Tests for Free-Wilson SAR + Prometheus metrics endpoint (v2.38, v2.39)."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from molprop.features.freewilson import free_wilson
from molprop.serving.api import app

# ── Free-Wilson core logic ───────────────────────────────────────────────────


class TestFreeWilson:
    def test_invalid_core_returns_none(self):
        assert free_wilson("not a core ((", ["c1ccccc1"], [1.0]) is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            free_wilson("c1ccccc1", ["c1ccc(O)cc1"], [1.0, 2.0])

    def test_perfect_additive_recovery(self):
        """
        Build an exactly-additive synthetic dataset and confirm Free-Wilson
        recovers the contributions and intercept (R² ≈ 1, RMSE ≈ 0).
        """
        analogues = [
            "c1ccc(O)cc1",  # R = OH
            "c1ccc(N)cc1",  # R = NH2
            "c1ccc(C)cc1",  # R = CH3
            "c1ccc(F)cc1",  # R = F
        ]
        activities = [5.0, 6.0, 4.5, 7.0]
        result = free_wilson("c1ccccc1", analogues, activities)
        assert result is not None
        assert result.n_used >= 2
        # With one R-group and one occupant per analogue, model is exact
        assert result.r_squared > 0.99 - 1e-9
        assert result.rmse < 1e-6 or math.isclose(result.rmse, 0.0, abs_tol=1e-6)
        # Intercept + reference contribution should reproduce the reference activity
        # Each prediction must equal its observation
        for p in result.predictions:
            assert abs(p["y_observed"] - p["y_predicted"]) < 1e-6

    def test_min_occurrences_filter(self):
        analogues = ["c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C)cc1"]
        activities = [5.0, 6.0, 4.5]
        # min_occurrences=2 drops every singleton occupant ⇒ no extra features,
        # model is just an intercept (mean of activities)
        result = free_wilson("c1ccccc1", analogues, activities, min_occurrences=2)
        assert result is not None
        # At most one feature would be retained (the most frequent occupant
        # is the implicit reference, so all singletons are dropped).
        non_ref_contribs = [c for c in result.contributions if c.contribution != 0.0]
        assert len(non_ref_contribs) <= 1


# ── /metrics/prometheus endpoint ─────────────────────────────────────────────


class TestPrometheusEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_prometheus_format(self):
        # First make a couple of requests so counters increment
        self.client.get("/health")
        self.client.get("/version")

        r = self.client.get("/metrics/prometheus")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]

        body = r.text
        # All five core metrics must appear
        assert "molprop_uptime_seconds" in body
        assert "molprop_requests_total" in body
        assert "molprop_request_errors_total" in body
        assert "molprop_request_latency_seconds_sum" in body
        assert "molprop_cache_hits_total" in body
        # And HELP/TYPE lines
        assert "# HELP molprop_uptime_seconds" in body
        assert "# TYPE molprop_uptime_seconds gauge" in body

    def test_prometheus_label_escaping_safe(self):
        r = self.client.get("/metrics/prometheus").text
        # No unescaped quotes inside label values
        for line in r.splitlines():
            if "{" not in line:
                continue
            # crude check: balanced quotes within {…}
            inside = line.split("{", 1)[1].rsplit("}", 1)[0]
            assert inside.count('"') % 2 == 0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
