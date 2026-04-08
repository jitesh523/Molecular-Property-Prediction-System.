import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from omegaconf import DictConfig
from rich.console import Console
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from molprop.data.splits import random_scaffold_split
from molprop.features.graphs import smiles_to_graph
from molprop.models.gnn_gat import GATModel
from molprop.models.gnn_gcn import GCNModel
from molprop.models.gnn_mpnn import MPNNModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
console = Console()


def train_one_epoch(model, loader, optimizer, device, task_type):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        # Pull labels
        y = data.y.view(-1, 1)

        if task_type == "regression":
            loss = F.mse_loss(out, y)
        else:
            # Classification
            loss = F.binary_cross_entropy_with_logits(out, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, task_type):
    model.eval()
    y_true_list = []
    y_pred_list = []

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

    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    import numpy as np
    from sklearn.metrics import mean_squared_error, roc_auc_score

    if task_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)  # RMSE
    else:
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            # Handle cases where only one class is present in a batch/split
            return 0.5


def run_training(cfg: DictConfig):
    # Device Setup
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

    # MLflow Setup
    # Use absolute path for tracking URI to avoid confusion with Hydra's changing CWD
    root_dir = Path(__file__).resolve().parent.parent
    tracking_uri = f"file://{root_dir}/{cfg.mlflow.tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Data Loading
    data_path = root_dir / "data" / "processed" / cfg.dataset.name / "processed.csv"
    if not data_path.exists():
        log.error(f"Data path not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    log.info(f"Training on {cfg.dataset.name} task: {cfg.dataset.target}")

    # Convert to Graphs
    log.info("Converting SMILES to Graphs...")
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        g = smiles_to_graph(row["std_smiles"], y=row[cfg.dataset.target])
        if g:
            dataset.append(g)

    if not dataset:
        log.error("No valid graphs created. Check connectivity or features.")
        return

    # Split
    smiles_list = [g.smiles for g in dataset]
    train_idx, val_idx, test_idx = random_scaffold_split(smiles_list)

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.training.batch_size)
    test_loader = DataLoader(test_data, batch_size=cfg.training.batch_size)

    # Model Initialization
    in_dim = dataset[0].num_node_features
    if cfg.model.type == "gcn":
        model = GCNModel(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=1,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.type == "gat":
        model = GATModel(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=1,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            heads=cfg.model.heads,
        )
    else:
        model = MPNNModel(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=1,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            edge_dim=cfg.model.edge_dim,
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    run_name = cfg.mlflow.run_name
    with mlflow.start_run(run_name=run_name):
        # Log Hyperparameters with prefixes to avoid collisions (e.g., 'name')
        mlflow.log_params({f"train.{k}": v for k, v in cfg.training.items()})
        mlflow.log_params({f"model.{k}": v for k, v in cfg.model.items()})
        mlflow.log_params({f"dataset.{k}": v for k, v in cfg.dataset.items()})

        # Training Loop
        best_val = float("inf") if cfg.dataset.task_type == "regression" else 0
        patience = cfg.training.patience
        counter = 0
        best_model_path = root_dir / f"best_model_{cfg.model.name}_{cfg.dataset.name}.pt"

        log.info(f"Starting Training ({cfg.model.name})...")
        for epoch in range(1, cfg.training.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device, cfg.dataset.task_type)
            val_score = evaluate(model, val_loader, device, cfg.dataset.task_type)

            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_score", val_score, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                log.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Score: {val_score:.4f}")

            # Early Stopping logic
            improved = False
            if cfg.dataset.task_type == "regression":
                if val_score < best_val:
                    best_val = val_score
                    improved = True
            else:
                if val_score > best_val:
                    best_val = val_score
                    improved = True

            if improved:
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(str(best_model_path))
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                log.info("Early stopping triggered.")
                break

        # Final Evaluation
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_score = evaluate(model, test_loader, device, cfg.dataset.task_type)
            log.info(f"Final Test Score: {test_score:.4f}")
            mlflow.log_metric("test_score", test_score)
        else:
            log.warning("No best model found. Skipping final evaluation.")


def main():
    # Use Hydra Compose API to avoid Python 3.14 argparse issues
    # The config_path is relative to the directory of this script, but initialize handles it
    with initialize(version_base=None, config_path="../configs"):
        # Take overrides from sys.argv
        overrides = sys.argv[1:]
        cfg = compose(config_name="config", overrides=overrides)
        run_training(cfg)


if __name__ == "__main__":
    main()
