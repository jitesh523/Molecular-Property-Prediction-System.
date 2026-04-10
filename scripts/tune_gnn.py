"""
Optuna hyperparameter tuning for GNN models (GCN, GAT, MPNN).
Integrates with Hydra config overrides and MLflow tracking.
"""

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from molprop.data.splits import random_scaffold_split
from molprop.features.graphs import smiles_to_graph
from molprop.models.gnn_gat import GATModel
from molprop.models.gnn_gcn import GCNModel
from molprop.models.gnn_mpnn import MPNNModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def build_model(trial, model_type, in_dim):
    """Create a GNN model with Optuna-sampled hyperparameters."""
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    if model_type == "gcn":
        return GCNModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "gat":
        heads = trial.suggest_categorical("heads", [2, 4, 8])
        return GATModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads,
        )
    else:  # mpnn
        return MPNNModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=4,
        )


def train_epoch(model, loader, optimizer, device, task_type):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.view(-1, 1)
        if task_type == "regression":
            loss = F.mse_loss(out, y)
        else:
            loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, device, task_type):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        y_true.append(data.y.view(-1, 1).cpu())
        if task_type == "classification":
            y_pred.append(torch.sigmoid(out).cpu())
        else:
            y_pred.append(out.cpu())

    if not y_true:
        return 0.0

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    from sklearn.metrics import mean_squared_error, roc_auc_score

    if task_type == "regression":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.5


def objective(trial, model_type, task_type, dataset, train_idx, val_idx, device, max_epochs=50):
    """Single Optuna trial for a GNN model."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    in_dim = dataset[0].num_node_features
    model = build_model(trial, model_type, in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    best_val = float("inf") if task_type == "regression" else 0
    patience_counter = 0
    patience = 10

    for epoch in range(1, max_epochs + 1):
        train_epoch(model, train_loader, optimizer, device, task_type)
        val_score = evaluate_model(model, val_loader, device, task_type)

        # Report for Optuna pruning
        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        improved = val_score < best_val if task_type == "regression" else val_score > best_val
        if improved:
            best_val = val_score
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_val


def run_gnn_tuning(
    dataset_name: str,
    task_type: str,
    model_type: str,
    processed_dir: Path,
    n_trials: int = 30,
    output_dir: Path = None,
):
    """Run Optuna tuning for a GNN model on a dataset."""
    data_path = processed_dir / dataset_name / "processed.csv"
    if not data_path.exists():
        log.error(f"Data not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    target_cols = [c for c in df.columns if c != "std_smiles"]
    target = target_cols[0]
    log.info(f"Tuning {model_type.upper()} GNN on {dataset_name} (target: {target})")

    # Build graph dataset
    log.info("Converting SMILES to graphs...")
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        g = smiles_to_graph(row["std_smiles"], y=row[target])
        if g:
            dataset.append(g)

    if not dataset:
        log.error("No valid graphs.")
        return

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    # Split
    smiles_list = [g.smiles for g in dataset]
    train_idx, val_idx, _ = random_scaffold_split(smiles_list)

    # MLflow
    root_dir = Path(__file__).resolve().parent.parent
    tracking_uri = f"file://{root_dir}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"HPO_{model_type}_{dataset_name}")

    # Optuna study
    direction = "minimize" if task_type == "regression" else "maximize"
    study = optuna.create_study(
        direction=direction,
        study_name=f"{model_type}_{dataset_name}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    def wrapped_objective(trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            score = objective(trial, model_type, task_type, dataset, train_idx, val_idx, device)
            mlflow.log_params(trial.params)
            metric_name = "val_rmse" if task_type == "regression" else "val_auc"
            mlflow.log_metric(metric_name, score)
            return score

    study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)

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
    parser = argparse.ArgumentParser(description="Optuna HPO for GNN models")
    parser.add_argument("--dataset", type=str, default="delaney")
    parser.add_argument(
        "--task", type=str, default="regression", choices=["regression", "classification"]
    )
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "mpnn"])
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    processed_path = ROOT / "data" / "processed"

    run_gnn_tuning(args.dataset, args.task, args.model, processed_path, n_trials=args.n_trials)
