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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from molprop.data.splits import random_scaffold_split
from molprop.data.transformer_dataset import SMILESDataset
from molprop.models.transformer import SMILESTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
console = Console()


def train_one_epoch(model, loader, optimizer, device, task_type):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device).view(-1, 1)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask)

        if task_type == "regression":
            loss = F.mse_loss(out, y)
        else:
            loss = F.binary_cross_entropy_with_logits(out, y)

        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, device, task_type):
    model.eval()
    y_true_list = []
    y_pred_list = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].view(-1, 1).cpu()

        out = model(input_ids, attention_mask)

        y_true_list.append(y)
        if task_type == "classification":
            y_pred_list.append(torch.sigmoid(out).cpu())
        else:
            y_pred_list.append(out.cpu())

    if not y_true_list:
        return 0.0

    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    if task_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)
    else:
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
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

    # HF Tokenizer
    log.info(f"Loading Tokenizer: {cfg.model.hf_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_model_name)

    # Dataset splits
    smiles_list = df["std_smiles"].tolist()
    train_idx, val_idx, test_idx = random_scaffold_split(smiles_list)

    # Sub-dataframe mapping
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    max_length = cfg.model.get("max_length", 128)

    # We pass 'std_smiles' and rename it inside the Dataset structure logic?
    # Wait, our dataset class expects "smiles". We will rename std_smiles to smiles before passing
    train_df = train_df.rename(columns={"std_smiles": "smiles"})
    val_df = val_df.rename(columns={"std_smiles": "smiles"})
    test_df = test_df.rename(columns={"std_smiles": "smiles"})

    train_dataset = SMILESDataset(train_df, [cfg.dataset.target], tokenizer, max_length)
    val_dataset = SMILESDataset(val_df, [cfg.dataset.target], tokenizer, max_length)
    test_dataset = SMILESDataset(test_df, [cfg.dataset.target], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.model.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.model.batch_size)

    # Model
    model = SMILESTransformer(
        model_name=cfg.model.hf_model_name, num_tasks=1, dropout=cfg.model.dropout
    ).to(device)

    # Transformers learning rates
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.model.learning_rate, weight_decay=cfg.model.weight_decay
    )

    run_name = f"{cfg.model.name}_{cfg.dataset.name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({f"model.{k}": v for k, v in cfg.model.items()})
        mlflow.log_params({f"dataset.{k}": v for k, v in cfg.dataset.items()})

        best_val = float("inf") if cfg.dataset.task_type == "regression" else 0
        patience = cfg.model.patience
        counter = 0
        best_model_path = root_dir / f"best_model_{cfg.model.name}_{cfg.dataset.name}.pt"

        log.info(f"Starting Training ({cfg.model.name})...")
        for epoch in range(1, cfg.model.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device, cfg.dataset.task_type)
            val_score = evaluate(model, val_loader, device, cfg.dataset.task_type)

            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_score", val_score, step=epoch)

            if epoch % 1 == 0:
                log.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Score: {val_score:.4f}")

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
    with initialize(version_base=None, config_path="../configs"):
        overrides = sys.argv[1:]
        cfg = compose(config_name="config", overrides=overrides)
        run_training(cfg)


if __name__ == "__main__":
    main()
