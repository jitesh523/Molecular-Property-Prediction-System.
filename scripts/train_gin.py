"""
Train a GIN (Graph Isomorphism Network) model on a molecular property dataset.

Extends the base GNN training pipeline with GIN-specific defaults:
  - 4 message-passing layers (vs. 3 for GCN/GAT)
  - JK-sum multi-scale readout (built into GINModel)
  - Per-layer BatchNorm for training stability
  - AdamW optimizer with cosine-annealing LR schedule

Usage:
    python scripts/train_gin.py dataset.name=bbbp dataset.target=p_np
    python scripts/train_gin.py dataset.name=delaney dataset.target=measured_log_solubility_in_mols_per_litre
"""

import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import DictConfig
from rich.console import Console
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from molprop.data.splits import random_scaffold_split
from molprop.features.graphs import smiles_to_graph
from molprop.models.evaluate import compute_metrics
from molprop.models.gnn_gin import GINModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
console = Console()


def train_one_epoch(model, loader, optimizer, device, task_type):
    model.train()
    total_loss = 0.0
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, task_type):
    model.eval()
    y_true_list, y_pred_list = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        y_true_list.append(data.y.view(-1, 1).cpu())
        if task_type == "classification":
            y_pred_list.append(torch.sigmoid(out).cpu())
        else:
            y_pred_list.append(out.cpu())

    if not y_true_list:
        return 0.0

    y_true = torch.cat(y_true_list).numpy().ravel()
    y_pred = torch.cat(y_pred_list).numpy().ravel()

    if task_type == "regression":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    else:
        try:
            return float(roc_auc_score(y_true, y_pred))
        except ValueError:
            return 0.5


def run_training(cfg: DictConfig):
    # ── Device ───────────────────────────────────────────────────────────────
    if cfg.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)
    log.info(f"Device: {device}")

    # ── MLflow ───────────────────────────────────────────────────────────────
    root_dir = Path(__file__).resolve().parent.parent
    tracking_uri = f"file://{root_dir}/{cfg.mlflow.tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # ── Data ─────────────────────────────────────────────────────────────────
    data_path = root_dir / "data" / "processed" / cfg.dataset.name / "processed.csv"
    if not data_path.exists():
        log.error(f"Processed data not found: {data_path}")
        log.error("Run `python scripts/download_molnet_datasets.py` first.")
        return

    df = pd.read_csv(data_path)
    log.info(f"Dataset: {cfg.dataset.name}  |  Target: {cfg.dataset.target}  |  N={len(df)}")

    log.info("Converting SMILES to graphs ...")
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        g = smiles_to_graph(row["std_smiles"], y=float(row[cfg.dataset.target]))
        if g:
            dataset.append(g)
    log.info(f"Valid graphs: {len(dataset)}")

    # ── Split ────────────────────────────────────────────────────────────────
    smiles_list = [g.smiles for g in dataset]
    train_idx, val_idx, test_idx = random_scaffold_split(smiles_list)

    train_loader = DataLoader(
        [dataset[i] for i in train_idx], batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=cfg.training.batch_size)
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=cfg.training.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    in_dim = dataset[0].num_node_features
    model = GINModel(
        in_dim=in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=1,
        num_layers=4,
        dropout=cfg.model.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"GIN parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-5)

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val = float("inf") if cfg.dataset.task_type == "regression" else 0.0
    patience_counter = 0
    best_model_path = root_dir / f"best_model_gin_{cfg.dataset.name}.pt"

    run_name = f"gin_{cfg.dataset.name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": "gin",
                "num_layers": 4,
                "hidden_dim": cfg.model.hidden_dim,
                "dropout": cfg.model.dropout,
                "lr": cfg.training.lr,
                "batch_size": cfg.training.batch_size,
                "dataset": cfg.dataset.name,
                "task_type": cfg.dataset.task_type,
            }
        )

        log.info("Starting GIN training ...")
        for epoch in range(1, cfg.training.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device, cfg.dataset.task_type)
            val_score = evaluate(model, val_loader, device, cfg.dataset.task_type)
            scheduler.step()

            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_score", val_score, step=epoch)
            mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    f"Epoch {epoch:03d}  loss={loss:.4f}  val={val_score:.4f}"
                    f"  lr={scheduler.get_last_lr()[0]:.2e}"
                )

            improved = (
                val_score < best_val
                if cfg.dataset.task_type == "regression"
                else val_score > best_val
            )
            if improved:
                best_val = val_score
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(str(best_model_path))
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.training.patience:
                log.info(f"Early stopping at epoch {epoch}.")
                break

        # ── Final Evaluation ──────────────────────────────────────────────────
        if best_model_path.exists():
            model.load_state_dict(
                torch.load(best_model_path, map_location=device, weights_only=True)
            )
            evaluate(model, test_loader, device, cfg.dataset.task_type)

            y_true_list, y_pred_list = [], []
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data)
                    y_true_list.append(data.y.view(-1).cpu().numpy())
                    if cfg.dataset.task_type == "classification":
                        y_pred_list.append(torch.sigmoid(out).view(-1).cpu().numpy())
                    else:
                        y_pred_list.append(out.view(-1).cpu().numpy())

            y_true = np.concatenate(y_true_list)
            y_pred = np.concatenate(y_pred_list)
            metrics = compute_metrics(y_true, y_pred, task=cfg.dataset.task_type)

            log.info(f"\nTest results for GIN on {cfg.dataset.name}:")
            for k, v in metrics.items():
                log.info(f"  {k}: {v}")
                mlflow.log_metric(f"test_{k}", v)
        else:
            log.warning("No best model checkpoint found.")


def main():
    with initialize(version_base=None, config_path="../configs"):
        overrides = sys.argv[1:]
        cfg = compose(config_name="config", overrides=overrides)
        run_training(cfg)


if __name__ == "__main__":
    main()
