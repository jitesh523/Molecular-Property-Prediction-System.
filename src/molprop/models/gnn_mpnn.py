import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from molprop.models.gnn_base import GNNBase


class MPNNLayer(MessagePassing):
    """
    Message Passing Layer that incorporates edge features into the message.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super(MPNNLayer, self).__init__(aggr="add")
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: node features for neighbors
        # edge_attr: edge features for connectivity
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        # aggr_out: aggregated messages
        # x: current node features
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class MPNNModel(GNNBase):
    """
    Message Passing Neural Network (MPNN) for molecules.
    """

    def __init__(self, edge_dim: int = 3, **kwargs):
        super(MPNNModel, self).__init__(**kwargs)
        self.edge_dim = edge_dim
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(MPNNLayer(self.in_dim, self.hidden_dim, self.edge_dim))

        # Intermediate layers
        for _ in range(self.num_layers - 1):
            self.layers.append(MPNNLayer(self.hidden_dim, self.hidden_dim, self.edge_dim))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Message Passing with Edge Features
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global Readout
        x = self.pooling(x, batch)

        # Prediction Head
        return self.mlp(x)
