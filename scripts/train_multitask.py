"""
Multi-task GNN training script.

Trains a shared GNN backbone with per-task prediction heads on datasets
with multiple target columns. Supports NaN-masked loss for incomplete
label matrices (common in multi-assay pharmaceutical data).

Usage:
    python scripts/train_multitask.py model=multitask dataset=bbbp
"""

import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from rich.console import Console
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from molprop.data.splits import random_scaffold_split
from molprop.features.graphs import smiles_to_graph
from molprop.models.gnn_multitask import MultiTaskGNN, masked_multitask_loss

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
console = Console()


def train_one_epoch(model, loader, optimizer, device, task_types):
    """Train for one epoch with masked multi-task loss."""
    model.train()
    total_loss = 0
    n_graphs = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        y = data.y  # (batch, num_tasks) — may contain NaN
        loss = masked_multitask_loss(out, y, task_types)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        n_graphs += data.num_graphs

    return total_loss / max(n_graphs, 1)


@torch.no_grad()
def evaluate_multitask(model, loader, device, task_types, task_names):
    """Evaluate per-task metrics, skipping NaN entries."""
    model.eval()
    y_true_all, y_pred_all = [], []

    for data in loader:
        data = data.to(device)
        out = model(data)
        y_true_all.append(data.y.cpu().numpy())
        y_pred_all.append(out.cpu().numpy())

    if not y_true_all:
        return {}

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    metrics = {}
    for i, (tt, name) in enumerate(zip(task_types, task_names, strict=False)):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() < 5:
            continue  # Too few valid labels for meaningful evaluation

        yt = y_true[mask, i]
        yp = y_pred[mask, i]

        if tt == "regression":
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            metrics[f"{name}_rmse"] = rmse
        else:
            try:
                yp_prob = 1.0 / (1.0 + np.exp(-yp))  # sigmoid
                auc = float(roc_auc_score(yt, yp_prob))
            except ValueError:
                auc = 0.5
            metrics[f"{name}_roc_auc"] = auc

    return metrics


def run_training(cfg: DictConfig):
    """Main training loop for multi-task GNN."""
    # Device
    if cfg.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    # Paths
    root_dir = Path(__file__).resolve().parent.parent
    tracking_uri = f"file://{root_dir}/{cfg.mlflow.tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Data
    data_path = root_dir / "data" / "processed" / cfg.dataset.name / "processed.csv"
    if not data_path.exists():
        log.error(f"Data path not found: {data_path}")
        return

    df = pd.read_csv(data_path)

    # Identify target columns (everything except std_smiles)
    target_cols = [c for c in df.columns if c != "std_smiles"]
    num_tasks = len(target_cols)
    log.info(f"Multi-task training on {cfg.dataset.name}: {num_tasks} tasks → {target_cols}")

    # Determine task types from config or infer
    task_types = cfg.model.get("task_types", None)
    if task_types:
        task_types = list(task_types)
    else:
        # Infer: if all values are 0/1/NaN → classification, else regression
        task_types = []
        for col in target_cols:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                task_types.append("classification")
            else:
                task_types.append("regression")
        log.info(f"Inferred task types: {dict(zip(target_cols, task_types, strict=False))}")

    # Build graphs with multi-task labels
    log.info("Converting SMILES to graphs...")
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        g = smiles_to_graph(row["std_smiles"])
        if g is None:
            continue
        # Build multi-task label tensor (NaN preserved)
        y_vals = [float(row[c]) if pd.notna(row[c]) else float("nan") for c in target_cols]
        g.y = torch.tensor([y_vals], dtype=torch.float)  # (1, num_tasks)
        dataset.append(g)

    if not dataset:
        log.error("No valid graphs. Aborting.")
        return

    # Scaffold split
    smiles_list = [g.smiles for g in dataset]
    train_idx, val_idx, test_idx = random_scaffold_split(smiles_list)

    train_loader = DataLoader(
        [dataset[i] for i in train_idx],
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=cfg.training.batch_size)
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=cfg.training.batch_size)

    # Model
    in_dim = dataset[0].num_node_features
    backbone = cfg.model.get("backbone", "gcn")
    backbone_kwargs = {}
    if backbone == "gat":
        backbone_kwargs["heads"] = cfg.model.get("heads", 4)
    elif backbone == "mpnn":
        backbone_kwargs["edge_dim"] = cfg.model.get("edge_dim", 3)

    model = MultiTaskGNN(
        backbone=backbone,
        in_dim=in_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_tasks=num_tasks,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        task_types=task_types,
        **backbone_kwargs,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.get("weight_decay", 1e-4),
    )

    run_name = cfg.mlflow.get("run_name", f"multitask_{backbone}_{cfg.dataset.name}")
    best_model_path = root_dir / f"best_multitask_{backbone}_{cfg.dataset.name}.pt"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "backbone": backbone,
                "num_tasks": num_tasks,
                "task_types": str(task_types),
                "hidden_dim": cfg.model.hidden_dim,
                "num_layers": cfg.model.num_layers,
                "dropout": cfg.model.dropout,
                "lr": cfg.training.lr,
                "batch_size": cfg.training.batch_size,
                "dataset": cfg.dataset.name,
            }
        )

        best_val_score = float("inf")
        patience_counter = 0
        patience = cfg.training.patience

        log.info(f"Starting multi-task training ({backbone}, {num_tasks} tasks)...")
        for epoch in range(1, cfg.training.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device, task_types)
            val_metrics = evaluate_multitask(model, val_loader, device, task_types, target_cols)

            mlflow.log_metric("train_loss", loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

            # Aggregate validation score for early stopping
            if val_metrics:
                val_score = np.mean(list(val_metrics.values()))
            else:
                val_score = loss

            if epoch % 10 == 0 or epoch == 1:
                metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                log.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | {metrics_str}")

            # Early stopping (minimizing for regression, but we use loss as proxy)
            if val_score < best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(str(best_model_path))
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                log.info("Early stopping triggered.")
                break

        # Final test evaluation
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_metrics = evaluate_multitask(model, test_loader, device, task_types, target_cols)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())
            log.info(f"Test Metrics: {metrics_str}")

    console.print("\n[bold green]Multi-task training complete![/bold green]")
    console.print(f"Best model saved to: {best_model_path}")


def main():
    with initialize(version_base=None, config_path="../configs"):
        overrides = sys.argv[1:]
        cfg = compose(config_name="config", overrides=overrides)
        run_training(cfg)


if __name__ == "__main__":
    main()
