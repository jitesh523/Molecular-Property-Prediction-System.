import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from molprop.models.gnn_base import GNNBase


class GCNModel(GNNBase):
    """
    Graph Convolutional Network (GCN) for molecules.
    """

    def __init__(self, **kwargs):
        super(GCNModel, self).__init__(**kwargs)
        self.convs = torch.nn.ModuleList()
        # First layer
        self.convs.append(GCNConv(self.in_dim, self.hidden_dim))
        # Subsequent layers
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

    def forward(self, data, mc_dropout: bool = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        is_training = self.training or mc_dropout

        # Message Passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=is_training)

        # Global Readout
        x = self.pooling(x, batch)

        # Prediction Head
        return self.mlp(x)

    @torch.no_grad()
    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return self.pooling(x, batch)
