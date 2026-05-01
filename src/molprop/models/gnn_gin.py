import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GINConv

from molprop.models.gnn_base import GNNBase


class GINModel(GNNBase):
    """
    Graph Isomorphism Network (GIN) for molecular property prediction.

    GIN is provably the most expressive GNN in the WL-test hierarchy,
    making it well-suited for distinguishing non-isomorphic molecular graphs.
    Reference: Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019.
    """

    def __init__(self, **kwargs):
        super(GINModel, self).__init__(**kwargs)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        dims = [self.in_dim] + [self.hidden_dim] * self.num_layers

        for i in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], dims[i + 1] * 2),
                nn.BatchNorm1d(dims[i + 1] * 2),
                nn.ReLU(),
                nn.Linear(dims[i + 1] * 2, dims[i + 1]),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm(dims[i + 1]))

    def forward(self, data, mc_dropout: bool = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        is_training = self.training or mc_dropout

        node_embeddings = []
        for conv, bn in zip(self.convs, self.batch_norms, strict=False):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=is_training)
            node_embeddings.append(x)

        # Sum over all layer embeddings (JK-sum / multi-scale readout)
        x = sum(self.pooling(h, batch) for h in node_embeddings)

        return self.mlp(x)

    @torch.no_grad()
    def encode(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_embeddings = []
        for conv, bn in zip(self.convs, self.batch_norms, strict=False):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            node_embeddings.append(x)
        return sum(self.pooling(h, batch) for h in node_embeddings)
