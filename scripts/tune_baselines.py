"""
Optuna hyperparameter tuning for baseline models (RF, XGBoost).
Logs each trial to MLflow and saves the best config as JSON.
"""

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from molprop.data.splits import random_scaffold_split
from molprop.features.fingerprints import batch_smiles_to_morgan

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def create_rf_model(trial, task_type):
    params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("rf_max_depth", 5, 50),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
        "random_state": 42,
        "n_jobs": -1,
    }
    if task_type == "classification":
        params["class_weight"] = "balanced"
        return RandomForestClassifier(**params), params
    return RandomForestRegressor(**params), params


def create_xgb_model(trial, task_type):
    params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 2000, step=100),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.01, 10.0, log=True),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.001, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }
    if task_type == "classification":
        return XGBClassifier(**params, eval_metric="logloss"), params
    return XGBRegressor(**params), params


def objective(trial, model_type, task_type, x_train, y_train, x_val, y_val):
    """Single Optuna trial: build model, train, evaluate."""
    if model_type == "rf":
        model, params = create_rf_model(trial, task_type)
    else:
        model, params = create_xgb_model(trial, task_type)

    model.fit(x_train, y_train)

    if task_type == "regression":
        y_pred = model.predict(x_val)
        score = np.sqrt(mean_squared_error(y_val, y_pred))
        # For regression, lower RMSE is better → Optuna minimizes
        return score
    else:
        y_proba = model.predict_proba(x_val)
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_score = y_proba[:, 1]
        else:
            y_score = y_proba
        try:
            score = roc_auc_score(y_val, y_score)
        except ValueError:
            score = 0.5
        # For classification, higher AUC is better → Optuna maximizes
        return score


def run_tuning(
    dataset_name: str,
    task_type: str,
    model_type: str,
    processed_dir: Path,
    n_trials: int = 50,
    output_dir: Path = None,
):
    """Run Optuna tuning for a single model/dataset combo."""
    data_path = processed_dir / dataset_name / "processed.csv"
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    smiles_list = df["std_smiles"].tolist()
    target_cols = [c for c in df.columns if c != "std_smiles"]
    target = target_cols[0]

    log.info(f"Tuning {model_type.upper()} on {dataset_name} (target: {target})")

    # Featurize
    x = batch_smiles_to_morgan(smiles_list)
    y = df[target].values

    valid_mask = ~pd.isna(y)
    x = x[valid_mask]
    y = y[valid_mask]
    smiles_valid = [smiles_list[i] for i in range(len(smiles_list)) if valid_mask[i]]

    # Split
    train_idx, val_idx, _ = random_scaffold_split(smiles_valid)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    # Optuna Study
    direction = "minimize" if task_type == "regression" else "maximize"
    study = optuna.create_study(
        direction=direction,
        study_name=f"{model_type}_{dataset_name}",
        pruner=optuna.pruners.MedianPruner(),
    )

    # MLflow tracking
    root_dir = Path(__file__).resolve().parent.parent
    tracking_uri = f"file://{root_dir}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"HPO_{model_type}_{dataset_name}")

    def wrapped_objective(trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            score = objective(trial, model_type, task_type, x_train, y_train, x_val, y_val)
            mlflow.log_params(trial.params)
            metric_name = "val_rmse" if task_type == "regression" else "val_auc"
            mlflow.log_metric(metric_name, score)
            return score

    study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)

    # Save best config
    log.info(f"Best trial: {study.best_trial.number}")
    log.info(f"Best value: {study.best_value:.4f}")
    log.info(f"Best params: {study.best_params}")

    if output_dir is None:
        output_dir = root_dir / "results" / "hpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / f"best_{model_type}_{dataset_name}.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "model_type": model_type,
                "dataset": dataset_name,
                "task_type": task_type,
                "best_value": study.best_value,
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )
    log.info(f"Best config saved to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for baseline models")
    parser.add_argument("--dataset", type=str, default="delaney")
    parser.add_argument(
        "--task", type=str, default="regression", choices=["regression", "classification"]
    )
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "xgb"])
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    processed_path = ROOT / "data" / "processed"

    run_tuning(args.dataset, args.task, args.model, processed_path, n_trials=args.n_trials)
