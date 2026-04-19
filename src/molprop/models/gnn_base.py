import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool


class GNNBase(nn.Module):
    """
    Base class for GNN models in the molecular property prediction system.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        readout: str = "mean",
    ):
        super(GNNBase, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout = readout

        # MLP Head for prediction after graph pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim),
        )

    def pooling(self, x, batch):
        if self.readout == "mean":
            return global_mean_pool(x, batch)
        elif self.readout == "max":
            return global_max_pool(x, batch)
        elif self.readout == "sum":
            from torch_geometric.nn import global_add_pool

            return global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown readout method: {self.readout}")

    def forward(self, data):
        raise NotImplementedError("Subclasses must implement the forward pass.")

    @torch.no_grad()
    def encode(self, data):
        """
        Extract the latent embedding (bottleneck) for vector search.
        """
        self.eval()
        raise NotImplementedError("Subclasses must implement the encode pass.")

    def get_device(self):
        """Helper to get the model's device."""
        return next(self.parameters()).device
