from typing import Any, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer


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


def get_explainer(
    model: torch.nn.Module, 
    task_type: str = "binary_classification",
    algorithm: str = "gnn"
) -> Explainer:
    """
    Sets up a PyG Explainer for the given model.
    algorithm: 'gnn' (GNNExplainer) or 'captum' (IntegratedGradients).
    task_type can be 'binary_classification', 'multiclass_classification', or 'regression'.
    """
    wrapper = ModelWrapper(model)
    
    if algorithm == "gnn":
        algo = GNNExplainer(epochs=200)
    elif algorithm == "captum":
        algo = CaptumExplainer("IntegratedGradients")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return Explainer(
        model=wrapper,
        algorithm=algo,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode=task_type,
            task_level="graph",
            return_type="raw",
        ),
    )


def explain_graph(
    explainer: Explainer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
) -> Any:
    """
    Generates an Explanation object containing node and edge masks for a single graph.
    """
    kwargs = {}
    if edge_attr is not None:
        kwargs["edge_attr"] = edge_attr

    explanation = explainer(x, edge_index, **kwargs)
    return explanation
