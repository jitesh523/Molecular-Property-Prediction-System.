import logging
from typing import Dict, Literal

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
