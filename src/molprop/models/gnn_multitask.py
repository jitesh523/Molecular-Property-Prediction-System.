"""
Multi-Task GNN for molecular property prediction.

Wraps any GNN backbone (GCN, GAT, MPNN) with a multi-output prediction head
and provides NaN-masked loss functions so missing labels don't contribute
gradients. This is critical for realistic pharma datasets (e.g. Tox21)
where assay completeness varies across molecules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from molprop.models.gnn_gat import GATModel
from molprop.models.gnn_gcn import GCNModel
from molprop.models.gnn_mpnn import MPNNModel


class MultiTaskGNN(nn.Module):
    """
    Multi-task GNN that shares a GNN backbone and splits into
    per-task prediction heads.

    Supports mixed task types (regression + classification) by
    applying appropriate losses per task column.
    """

    def __init__(
        self,
        backbone: str = "gcn",
        in_dim: int = 9,
        hidden_dim: int = 128,
        num_tasks: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
        task_types: list | None = None,
        **backbone_kwargs,
    ):
        """
        Args:
            backbone: One of 'gcn', 'gat', 'mpnn'.
            in_dim: Input node feature dimension.
            hidden_dim: Hidden layer dimension in backbone.
            num_tasks: Number of output tasks.
            num_layers: Number of message passing layers.
            dropout: Dropout probability.
            task_types: List of 'regression' or 'classification' per task.
                        Defaults to all 'regression' if not provided.
            **backbone_kwargs: Extra kwargs passed to the backbone constructor
                               (e.g. heads for GAT, edge_dim for MPNN).
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.task_types = task_types or ["regression"] * num_tasks

        # Build backbone — we use out_dim=hidden_dim to get embeddings,
        # then attach our own multi-task heads on top.
        common = dict(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,  # embedding output, NOT final prediction
            num_layers=num_layers,
            dropout=dropout,
        )

        if backbone == "gcn":
            self.backbone = GCNModel(**common)
        elif backbone == "gat":
            self.backbone = GATModel(heads=backbone_kwargs.get("heads", 4), **common)
        elif backbone == "mpnn":
            self.backbone = MPNNModel(edge_dim=backbone_kwargs.get("edge_dim", 3), **common)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Override the backbone's MLP head — we replace it with identity
        # so forward() returns the graph-level embedding.
        self.backbone.mlp = nn.Identity()

        # Multi-task prediction heads (one 2-layer MLP per task)
        self.task_heads = nn.ModuleList()
        for _ in range(num_tasks):
            self.task_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
            )

    def forward(self, data, mc_dropout: bool = False):
        """
        Returns:
            Tensor of shape (batch_size, num_tasks) with raw logits/values.
        """
        # Get graph-level embeddings from backbone
        # backbone is a GNNModel whose MLP was replaced by Identity
        # We need to manually apply dropout if mc_dropout is True because
        # backbone.mlp was Identity, but we can call it directly.
        emb = self.backbone(data, mc_dropout=mc_dropout)  # (batch_size, hidden_dim)

        # Apply each task head
        # task_heads is a list of Sequential: [Linear, ReLU, Dropout, Linear]
        # Sequential doesn't support passing args to intermediate layers,
        # so we manually implement the head logic to respect mc_dropout.
        is_training = self.training or mc_dropout
        outputs = []
        for head in self.task_heads:
            # head is Sequential(Linear, ReLU, Dropout, Linear)
            h = head[0](emb)
            h = head[1](h)
            h = F.dropout(h, p=self.backbone.dropout, training=is_training)
            h = head[3](h)
            outputs.append(h)

        return torch.cat(outputs, dim=-1)  # (batch_size, num_tasks)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE loss that ignores NaN entries in the target tensor.

    Args:
        pred: Predictions of shape (batch, num_tasks).
        target: Targets of shape (batch, num_tasks), may contain NaN.

    Returns:
        Scalar loss averaged over valid entries only.
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(pred[mask], target[mask])


def masked_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy with logits that ignores NaN entries.

    Args:
        pred: Raw logits of shape (batch, num_tasks).
        target: Binary targets of shape (batch, num_tasks), may contain NaN.

    Returns:
        Scalar loss averaged over valid entries only.
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(pred[mask], target[mask])


def masked_multitask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    task_types: list[str],
) -> torch.Tensor:
    """
    Combined loss for mixed regression + classification tasks.

    Applies MSE for regression columns and BCE-with-logits for
    classification columns, masking NaN in both cases.

    Args:
        pred: Shape (batch, num_tasks).
        target: Shape (batch, num_tasks), may contain NaN.
        task_types: List of 'regression' or 'classification' per task.
    """
    total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
    for i, tt in enumerate(task_types):
        p = pred[:, i : i + 1]
        t = target[:, i : i + 1]
        if tt == "regression":
            total_loss = total_loss + masked_mse_loss(p, t)
        else:
            total_loss = total_loss + masked_bce_loss(p, t)
    return total_loss / len(task_types)
