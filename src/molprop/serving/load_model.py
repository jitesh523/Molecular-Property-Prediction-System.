"""
Model loading utilities for the inference API.

Supports loading GNN models (GCN, GAT, MPNN, GIN) from saved state dicts
and baseline models (RF, XGBoost) from joblib artifacts.
"""

import logging
from pathlib import Path

import torch

from molprop.models.gnn_gat import GATModel
from molprop.models.gnn_gcn import GCNModel
from molprop.models.gnn_gin import GINModel
from molprop.models.gnn_mpnn import MPNNModel

log = logging.getLogger(__name__)

# Default architecture configs (match training defaults)
DEFAULT_GNN_CONFIGS = {
    "gcn": {
        "cls": GCNModel,
        "kwargs": {
            "hidden_dim": 128,
            "out_dim": 1,
            "num_layers": 3,
            "dropout": 0.2,
        },
    },
    "gat": {
        "cls": GATModel,
        "kwargs": {
            "hidden_dim": 128,
            "out_dim": 1,
            "num_layers": 3,
            "dropout": 0.2,
            "heads": 4,
        },
    },
    "mpnn": {
        "cls": MPNNModel,
        "kwargs": {
            "hidden_dim": 128,
            "out_dim": 1,
            "num_layers": 3,
            "dropout": 0.2,
            "edge_dim": 4,
        },
    },
    "gin": {
        "cls": GINModel,
        "kwargs": {
            "hidden_dim": 128,
            "out_dim": 1,
            "num_layers": 4,
            "dropout": 0.2,
        },
    },
}


def load_gnn_model(
    model_type: str,
    weights_path: str,
    in_dim: int = 9,
    hidden_dim: int = 128,
    out_dim: int = 1,
    num_layers: int = 3,
    dropout: float = 0.2,
    device: str = "cpu",
    **kwargs,
) -> torch.nn.Module:
    """
    Loads a predefined GNN model architecture and its weights.

    Args:
        model_type: One of 'gcn', 'gat', 'mpnn', 'gin'.
        weights_path: Path to the saved state dict.
        in_dim: Input node feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension (number of tasks).
        num_layers: Number of message passing layers.
        dropout: Dropout probability.
        device: Device to load model onto.
        **kwargs: Extra kwargs (e.g., heads for GAT, edge_dim for MPNN).
    """
    if model_type == "gcn":
        model = GCNModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "gat":
        heads = kwargs.get("heads", 4)
        model = GATModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads,
        )
    elif model_type == "mpnn":
        edge_dim = kwargs.get("edge_dim", 4)
        model = MPNNModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=edge_dim,
        )
    elif model_type == "gin":
        model = GINModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # nosec: B614 - loading local trusted weights
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    log.info(f"Loaded {model_type} model from {weights_path}")
    return model


def load_baseline_model(model_path: str):
    """
    Load a serialized baseline model (RF or XGBoost) via joblib.

    Args:
        model_path: Path to the joblib-serialized model.

    Returns:
        Deserialized model object.
    """
    import joblib

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline model not found: {model_path}")

    model = joblib.load(path)
    log.info(f"Loaded baseline model from {model_path}")
    return model
