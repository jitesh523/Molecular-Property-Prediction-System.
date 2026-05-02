"""
Tests for molprop.models.baselines.BaselineModel.

Covers RF and XGBoost for both classification and regression:
train/predict, evaluate, cross_validate, save/load, and
the feature_importances() method added in v1.5.0.
"""

import os
import tempfile

import numpy as np
import pytest

from molprop.models.baselines import BaselineModel

# ── Synthetic fixtures ─────────────────────────────────────────────────────────

RNG = np.random.default_rng(7)
N, D = 120, 15

X_CLF = RNG.standard_normal((N, D)).astype(np.float32)
Y_CLF = (X_CLF[:, 0] + X_CLF[:, 1] > 0).astype(int)

X_REG = RNG.standard_normal((N, D)).astype(np.float32)
Y_REG = (X_REG[:, 0] * 2.0 + X_REG[:, 1]).astype(np.float32)

FEATURE_NAMES = [f"fp_{i}" for i in range(D)]


# ── Classification: RF ────────────────────────────────────────────────────────


class TestRFClassification:
    @pytest.fixture
    def model(self):
        m = BaselineModel("rf", "classification", params={"n_estimators": 10, "random_state": 0})
        m.train(X_CLF, Y_CLF)
        return m

    def test_predict_shape(self, model):
        preds = model.predict(X_CLF)
        assert preds.shape == (N,)

    def test_predict_proba_shape(self, model):
        proba = model.predict_proba(X_CLF)
        assert proba.shape == (N, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_evaluate_keys(self, model):
        m = model.evaluate(X_CLF, Y_CLF)
        for key in ("roc_auc", "pr_auc", "mcc"):
            assert key in m

    def test_evaluate_roc_above_chance(self, model):
        m = model.evaluate(X_CLF, Y_CLF)
        assert m["roc_auc"] >= 0.5

    def test_feature_importances_length(self, model):
        fi = model.feature_importances()
        assert len(fi) == D

    def test_feature_importances_sorted_descending(self, model):
        fi = model.feature_importances()
        vals = list(fi.values())
        assert vals == sorted(vals, reverse=True)

    def test_feature_importances_with_names(self, model):
        fi = model.feature_importances(feature_names=FEATURE_NAMES)
        assert all(k.startswith("fp_") for k in fi)

    def test_feature_importances_default_names(self, model):
        fi = model.feature_importances()
        assert all(k.startswith("feature_") for k in fi)

    def test_feature_importances_sum_to_one(self, model):
        fi = model.feature_importances()
        assert sum(fi.values()) == pytest.approx(1.0, abs=1e-5)


# ── Classification: XGBoost ───────────────────────────────────────────────────


class TestXGBClassification:
    @pytest.fixture
    def model(self):
        m = BaselineModel(
            "xgb",
            "classification",
            params={"n_estimators": 10, "random_state": 0, "eval_metric": "logloss"},
        )
        m.train(X_CLF, Y_CLF)
        return m

    def test_predict_shape(self, model):
        assert model.predict(X_CLF).shape == (N,)

    def test_evaluate_has_roc_auc(self, model):
        m = model.evaluate(X_CLF, Y_CLF)
        assert "roc_auc" in m
        assert m["roc_auc"] >= 0.5

    def test_feature_importances_non_negative(self, model):
        fi = model.feature_importances()
        assert all(v >= 0.0 for v in fi.values())


# ── Regression: RF ────────────────────────────────────────────────────────────


class TestRFRegression:
    @pytest.fixture
    def model(self):
        m = BaselineModel("rf", "regression", params={"n_estimators": 10, "random_state": 0})
        m.train(X_REG, Y_REG)
        return m

    def test_predict_shape(self, model):
        assert model.predict(X_REG).shape == (N,)

    def test_evaluate_keys(self, model):
        m = model.evaluate(X_REG, Y_REG)
        for key in ("mae", "rmse", "r2"):
            assert key in m

    def test_evaluate_r2_positive(self, model):
        m = model.evaluate(X_REG, Y_REG)
        assert m["r2"] > 0.0

    def test_predict_proba_raises_for_regression(self, model):
        with pytest.raises(ValueError, match="predict_proba"):
            model.predict_proba(X_REG)

    def test_feature_importances_sorted(self, model):
        fi = model.feature_importances()
        vals = list(fi.values())
        assert vals == sorted(vals, reverse=True)


# ── Cross-validation ──────────────────────────────────────────────────────────


class TestCrossValidation:
    def test_clf_cv_fold_count(self):
        m = BaselineModel("rf", "classification", params={"n_estimators": 5, "random_state": 0})
        fold_metrics = m.cross_validate(X_CLF, Y_CLF, n_folds=3)
        for key in ("roc_auc", "pr_auc", "mcc"):
            assert len(fold_metrics[key]) == 3

    def test_reg_cv_fold_count(self):
        m = BaselineModel("rf", "regression", params={"n_estimators": 5, "random_state": 0})
        fold_metrics = m.cross_validate(X_REG, Y_REG, n_folds=3)
        for key in ("mae", "rmse", "r2"):
            assert len(fold_metrics[key]) == 3


# ── Save / Load ───────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_roundtrip_rf_classification(self):
        m = BaselineModel("rf", "classification", params={"n_estimators": 5, "random_state": 0})
        m.train(X_CLF, Y_CLF)
        preds_before = m.predict(X_CLF)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            m.save(path)
            loaded = BaselineModel.load(path, "rf", "classification")

        preds_after = loaded.predict(X_CLF)
        assert np.array_equal(preds_before, preds_after)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            BaselineModel.load("/nonexistent/path.joblib", "rf", "classification")


# ── Error handling ────────────────────────────────────────────────────────────


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="Unknown model type"):
        BaselineModel("svm", "classification")


def test_feature_importances_before_fit_raises():
    m = BaselineModel.__new__(BaselineModel)
    m.model_type = "rf"
    m.task_type = "classification"
    m.params = {}

    class _FakeModel:
        pass

    m.model = _FakeModel()
    with pytest.raises(AttributeError, match="feature_importances_"):
        m.feature_importances()
