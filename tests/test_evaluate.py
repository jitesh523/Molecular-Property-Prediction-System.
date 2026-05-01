"""
Tests for molprop.models.evaluate.compute_metrics.

Covers classification (perfect, random, threshold sensitivity, all keys)
and regression (perfect, near-perfect, all keys, invalid task) scenarios.
"""

import numpy as np
import pytest

from molprop.models.evaluate import compute_metrics

# ── Helpers ───────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)

CLASSIFICATION_KEYS = {
    "roc_auc",
    "avg_precision",
    "accuracy",
    "f1",
    "mcc",
    "specificity",
    "balanced_accuracy",
}

REGRESSION_KEYS = {
    "rmse",
    "mae",
    "r2",
    "mean_error",
    "std_error",
    "pearson_r",
}


# ── Classification ────────────────────────────────────────────────────────────


class TestComputeMetricsClassification:
    def test_perfect_classifier_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.05, 0.1, 0.9, 0.95])
        m = compute_metrics(y_true, y_score, task="classification")
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["mcc"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["balanced_accuracy"] == pytest.approx(1.0)

    def test_all_keys_present(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        m = compute_metrics(y_true, y_score, task="classification")
        assert CLASSIFICATION_KEYS.issubset(m.keys()), f"Missing: {CLASSIFICATION_KEYS - m.keys()}"

    def test_random_classifier_near_chance(self):
        y_true = RNG.integers(0, 2, size=400)
        y_score = RNG.uniform(0, 1, size=400)
        m = compute_metrics(y_true, y_score, task="classification")
        assert 0.3 < m["roc_auc"] < 0.7

    def test_custom_threshold_changes_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.4, 0.6, 0.4, 0.6])
        m_low = compute_metrics(y_true, y_score, task="classification", threshold=0.3)
        m_high = compute_metrics(y_true, y_score, task="classification", threshold=0.7)
        assert m_low["accuracy"] != m_high["accuracy"]

    def test_specificity_all_true_negatives(self):
        y_true = np.array([0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.1, 0.15])
        m = compute_metrics(y_true, y_score, task="classification")
        assert m["specificity"] == pytest.approx(1.0)

    def test_metric_values_are_floats(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7])
        m = compute_metrics(y_true, y_score, task="classification")
        for key, val in m.items():
            assert isinstance(val, float), f"{key} is not a float"


# ── Regression ────────────────────────────────────────────────────────────────


class TestComputeMetricsRegression:
    def test_perfect_regressor(self):
        y = np.linspace(0.0, 10.0, 50)
        m = compute_metrics(y, y, task="regression")
        assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert m["mae"] == pytest.approx(0.0, abs=1e-6)
        assert m["r2"] == pytest.approx(1.0, abs=1e-6)
        assert m["mean_error"] == pytest.approx(0.0, abs=1e-6)
        assert m["pearson_r"] == pytest.approx(1.0, abs=1e-6)

    def test_all_keys_present(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        m = compute_metrics(y_true, y_pred, task="regression")
        assert REGRESSION_KEYS.issubset(m.keys()), f"Missing: {REGRESSION_KEYS - m.keys()}"

    def test_pearson_r_range(self):
        y_true = np.linspace(0, 1, 100)
        y_pred = y_true + RNG.normal(0, 0.05, size=100)
        m = compute_metrics(y_true, y_pred, task="regression")
        assert -1.0 <= m["pearson_r"] <= 1.0

    def test_pearson_r_anticorrelated(self):
        y_true = np.linspace(0, 1, 50)
        y_pred = np.linspace(1, 0, 50)
        m = compute_metrics(y_true, y_pred, task="regression")
        assert m["pearson_r"] == pytest.approx(-1.0, abs=1e-6)

    def test_rmse_greater_than_mae(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
        y_pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        m = compute_metrics(y_true, y_pred, task="regression")
        assert m["rmse"] >= m["mae"]

    def test_metric_values_are_floats(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.0, 2.9])
        m = compute_metrics(y_true, y_pred, task="regression")
        for key, val in m.items():
            assert isinstance(val, float), f"{key} is not a float"


# ── Error handling ─────────────────────────────────────────────────────────────


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task type"):
        compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]), task="multiclass")
