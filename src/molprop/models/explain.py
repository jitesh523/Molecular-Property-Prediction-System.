import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from typing import Dict, Any, Optional

class ModelWrapper(torch.nn.Module):
    """
    Wraps models that take a 'data' object so they can be explained
    by PyTorch Geometric's Explainer which expects distinct tensors.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None, **kwargs):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data = Data(x=x, edge_index=edge_index, batch=batch, **kwargs)
        return self.model(data)

def get_explainer(model: torch.nn.Module, task_type: str = "binary_classification") -> Explainer:
    """
    Sets up a PyG Explainer for the given model using GNNExplainer.
    task_type can be 'binary_classification', 'multiclass_classification', or 'regression'.
    """
    wrapper = ModelWrapper(model)
    return Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode=task_type,
            task_level='graph',
            return_type='raw',
        ),
    )

def explain_graph(explainer: Explainer, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> Any:
    """
    Generates an Explanation object containing node and edge masks for a single graph.
    """
    kwargs = {}
    if edge_attr is not None:
        kwargs['edge_attr'] = edge_attr
        
    explanation = explainer(x, edge_index, **kwargs)
    return explanation
