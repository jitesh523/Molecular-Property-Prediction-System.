import torch

from molprop.models.gnn_gat import GATModel
from molprop.models.gnn_gcn import GCNModel
from molprop.models.gnn_mpnn import MPNNModel


def load_gnn_model(
    model_type: str,
    weights_path: str,
    in_dim: int,
    hidden_dim: int = 128,
    out_dim: int = 1,
    num_layers: int = 3,
    dropout: float = 0.2,
    device: str = "cpu",
    **kwargs,
) -> torch.nn.Module:
    """
    Loads a predefined GNN model architecture and its weights.
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
        edge_dim = kwargs.get("edge_dim", 4)  # fallback
        model = MPNNModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=edge_dim,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
