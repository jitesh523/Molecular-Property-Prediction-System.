import logging
from pathlib import Path
from typing import Dict, List, Literal

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier, XGBRegressor

log = logging.getLogger(__name__)

TaskType = Literal["regression", "classification"]


class BaselineModel:
    """
    Unified wrapper for baseline models (RF, XGBoost).
    """

    def __init__(
        self,
        model_type: Literal["rf", "xgb"],
        task_type: TaskType,
        params: Dict = None,
    ):
        self.model_type = model_type
        self.task_type = task_type
        self.params = params or {}
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == "rf":
            if self.task_type == "regression":
                return RandomForestRegressor(**self.params)
            else:
                return RandomForestClassifier(**self.params)
        elif self.model_type == "xgb":
            if self.task_type == "regression":
                return XGBRegressor(**self.params)
            else:
                return XGBClassifier(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.task_type == "classification":
            return self.model.predict_proba(x)
        else:
            raise ValueError("predict_proba is only available for classification.")

    def evaluate(self, x: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model and return a dictionary of metrics.
        """
        metrics = {}
        if self.task_type == "regression":
            y_pred = self.predict(x)
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["r2"] = float(r2_score(y_true, y_pred))
        else:
            # Classification
            y_pred = self.predict(x)
            y_proba = self.predict_proba(x)

            # Handle binary/multi-label classification
            if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                # Binary classification
                y_score = y_proba[:, 1]
            else:
                # Need to handle multi-label if necessary, but for now assuming binary or handled by scikit-learn
                y_score = y_proba

            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

        return metrics

    def save(self, path: str) -> None:
        """
        Serialize the fitted model to disk using joblib.

        Args:
            path: File path ending in '.joblib' or '.pkl'.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out)
        log.info(f"Saved {self.model_type} model to {out}")

    @classmethod
    def load(cls, path: str, model_type: str, task_type: TaskType) -> "BaselineModel":
        """
        Restore a previously saved BaselineModel from disk.

        Args:
            path: Path to the joblib artifact.
            model_type: 'rf' or 'xgb' (used for metadata only).
            task_type: 'regression' or 'classification'.

        Returns:
            A BaselineModel instance with the deserialized model.
        """
        obj = cls.__new__(cls)
        obj.model_type = model_type
        obj.task_type = task_type
        obj.params = {}
        obj.model = joblib.load(Path(path))
        log.info(f"Loaded {model_type} model from {path}")
        return obj

    def cross_validate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        seed: int = 42,
    ) -> Dict[str, List[float]]:
        """
        Run stratified k-fold (classification) or k-fold (regression) CV.

        Args:
            x: Feature matrix of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).
            n_folds: Number of CV folds.
            seed: Random seed for reproducibility.

        Returns:
            Dict mapping metric name → list of per-fold scores.
        """
        if self.task_type == "classification":
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        fold_metrics: Dict[str, List[float]] = {}
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
            x_tr, x_val = x[train_idx], x[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Reinitialize to avoid state bleed between folds
            fold_model = self.__class__(self.model_type, self.task_type, self.params)
            fold_model.train(x_tr, y_tr)
            metrics = fold_model.evaluate(x_val, y_val)

            for k, v in metrics.items():
                fold_metrics.setdefault(k, []).append(v)

            log.info(f"  Fold {fold_idx + 1}/{n_folds}: {metrics}")

        return fold_metrics
