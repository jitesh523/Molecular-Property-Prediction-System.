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
            self.convs.append(
                GATConv(
                    self.hidden_dim * self.heads, self.hidden_dim, heads=self.heads, concat=True
                )
            )

        # Final GAT layer (averaging or reducing heads before readout)
        self.convs.append(
            GATConv(self.hidden_dim * self.heads, self.hidden_dim, heads=1, concat=False)
        )

    def forward(self, data, mc_dropout: bool = False):
        """Forward pass for molecular property prediction using attention mechanism.
        
        Applies multi-head graph attention convolutions to learn adaptive edge
        weights between atoms. Reduces attention heads in final layer for readout.
        
        Args:
            data: PyG Data object with attributes:
                - x (Tensor): Node feature matrix of shape [num_nodes, in_dim]
                - edge_index (LongTensor): Edge indices [2, num_edges]
                - batch (LongTensor): Batch assignment vector [num_nodes]
            mc_dropout (bool): Enable Monte Carlo dropout for uncertainty estimation.
        
        Returns:
            Tensor: Predicted molecular properties of shape [batch_size, out_dim].
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        is_training = self.training or mc_dropout

        # Message Passing with Attention
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=is_training)

        # Global Readout
        x = self.pooling(x, batch)

        # Prediction Head
        return self.mlp(x)

    @torch.no_grad()
    def encode(self, data):
        """Extract latent graph embedding for similarity search.
        
        Generates molecular fingerprint by aggregating attention-weighted
        node representations across all graphs in the batch.
        
        Args:
            data: PyG Data object with same structure as forward().
        
        Returns:
            Tensor: Molecular embedding of shape [batch_size, hidden_dim].
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return self.pooling(x, batch)
