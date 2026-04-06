import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from molprop.models.gnn_base import GNNBase


class GATModel(GNNBase):
    """
    Graph Attention Network (GAT) for molecules.
    """

    def __init__(self, heads: int = 4, **kwargs):
        super(GATModel, self).__init__(**kwargs)
        self.heads = heads
        self.convs = torch.nn.ModuleList()

        # First layer (concatenating attention heads)
        self.convs.append(GATConv(self.in_dim, self.hidden_dim, heads=self.heads, concat=True))
        
        # Intermediate layers
        for _ in range(self.num_layers - 2):
            self.convs.append(GATConv(self.hidden_dim * self.heads, self.hidden_dim, heads=self.heads, concat=True))

        # Final GAT layer (averaging or reducing heads before readout)
        self.convs.append(GATConv(self.hidden_dim * self.heads, self.hidden_dim, heads=1, concat=False))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message Passing with Attention
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global Readout
        x = self.pooling(x, batch)

        # Prediction Head
        return self.mlp(x)
