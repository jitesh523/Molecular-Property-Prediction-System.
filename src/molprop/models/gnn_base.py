import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool


class GNNBase(nn.Module):
    """Base class for Graph Neural Network models in molecular property prediction.
    
    Provides common architecture components for GNN variants:
    - Configurable graph convolution layers with batch normalization
    - Multi-task readout (mean/max/sum pooling) for graph-level predictions
    - Flexible MLP prediction head
    - MC Dropout support for Bayesian uncertainty estimation
    - Latent encoding for vector similarity search
    
    Subclasses must implement:
    - forward(data, mc_dropout=False): Full prediction pass
    - encode(data): Embedding extraction pass
    
    Attributes:
        in_dim (int): Node feature dimension (typically 119 for atomic features)
        hidden_dim (int): Dimension of hidden graph representations
        out_dim (int): Output dimension (1 for regression, num_classes for classification)
        num_layers (int): Number of graph convolution layers
        dropout (float): Dropout probability for regularization
        readout (str): Graph pooling method ('mean', 'max', or 'sum')
        mlp (nn.Sequential): Final prediction MLP head
    
    Reference:
        - GCN: Kipf & Welling (ICLR 2017)
        - GAT: Velickovic et al. (ICLR 2018)
        - GIN: Xu et al. (ICLR 2019)
        - MPNN: Gilmer et al. (ICML 2017)
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
        """Aggregate node embeddings to graph-level representation.
        
        Applies permutation-invariant readout operation to convert per-node
        embeddings into a fixed-size graph-level vector suitable for the
        prediction MLP.
        
        Args:
            x (Tensor): Node embedding matrix [num_nodes, hidden_dim]
            batch (LongTensor): Batch assignment vector [num_nodes]
        
        Returns:
            Tensor: Aggregated embeddings [batch_size, hidden_dim]
        
        Raises:
            ValueError: If readout method is not supported.
        """
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
        """Predict molecular properties from graph representation.
        
        Must be implemented by subclasses with their specific convolution
        and aggregation strategies.
        
        Args:
            data: PyG Data object with x, edge_index, batch attributes
        
        Returns:
            Tensor: Predicted properties of shape [batch_size, out_dim]
        
        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError("Subclasses must implement the forward pass.")

    @torch.no_grad()
    def encode(self, data):
        """Extract latent molecular embedding for similarity search.
        
        Generates a fixed-size vector representation of the molecule that
        preserves structural and chemical information. Embeddings can be
        indexed in vector databases for fast k-NN retrieval of similar
        molecules or compounds with predicted similar properties.
        
        Must be implemented by subclasses to match their architecture.
        
        Args:
            data: PyG Data object with x, edge_index, batch attributes
        
        Returns:
            Tensor: Molecular embedding of shape [batch_size, hidden_dim]
        
        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        self.eval()
        raise NotImplementedError("Subclasses must implement the encode pass.")

    def get_device(self):
        """Helper to get the model's device."""
        return next(self.parameters()).device
