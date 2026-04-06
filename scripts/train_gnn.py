import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
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
            
    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()
    
    import numpy as np
    from sklearn.metrics import mean_squared_error, roc_auc_score
    
    if task_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse) # RMSE
    else:
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            # Handle cases where only one class is present in a batch/split
            return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="delaney", help="Dataset name")
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "mpnn"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    # Device Setup
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Using device: {device}")

    # Data Loading
    ROOT = Path(__file__).resolve().parent.parent
    data_path = ROOT / "data" / "processed" / args.dataset / "processed.csv"
    df = pd.read_csv(data_path)
    
    # Identify target column (assuming first non-smiles column)
    target_col = [c for c in df.columns if c != "std_smiles"][0]
    log.info(f"Training on {args.dataset} task: {target_col}")

    # Convert to Graphs
    log.info("Converting SMILES to Graphs...")
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        g = smiles_to_graph(row["std_smiles"], y=row[target_col])
        if g:
            dataset.append(g)

    # Split
    smiles_list = [g.smiles for g in dataset]
    train_idx, val_idx, test_idx = random_scaffold_split(smiles_list)
    
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # Model Initialization
    in_dim = dataset[0].num_node_features
    if args.model == "gcn":
        model = GCNModel(in_dim=in_dim, hidden_dim=args.hidden, out_dim=1, num_layers=args.layers)
    elif args.model == "gat":
        model = GATModel(in_dim=in_dim, hidden_dim=args.hidden, out_dim=1, num_layers=args.layers)
    else:
        model = MPNNModel(in_dim=in_dim, hidden_dim=args.hidden, out_dim=1, num_layers=args.layers)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    best_val = float("inf") if args.task == "regression" else 0
    patience = 20
    counter = 0

    log.info(f"Starting Training ({args.model})...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, args.task)
        val_score = evaluate(model, val_loader, device, args.task)
        
        if epoch % 10 == 0 or epoch == 1:
            log.info(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Score: {val_score:.4f}")

        # Early Stopping logic
        if args.task == "regression":
            if val_score < best_val:
                best_val = val_score
                torch.save(model.state_dict(), f"best_{args.model}_{args.dataset}.pt")
                counter = 0
            else:
                counter += 1
        else:
            if val_score > best_val:
                best_val = val_score
                torch.save(model.state_dict(), f"best_{args.model}_{args.dataset}.pt")
                counter = 0
            else:
                counter += 1

        if counter >= patience:
            log.info("Early stopping triggered.")
            break

    # Final Evaluation
    model.load_state_dict(torch.load(f"best_{args.model}_{args.dataset}.pt"))
    test_score = evaluate(model, test_loader, device, args.task)
    log.info(f"Final Test Score: {test_score:.4f}")


if __name__ == "__main__":
    main()
