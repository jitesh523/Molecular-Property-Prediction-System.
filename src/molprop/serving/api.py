import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from molprop.data.standardize import standardize_smiles
from molprop.features.graphs import smiles_to_graph
from molprop.serving.load_model import load_gnn_model
from molprop.models.explain import get_explainer, explain_graph

# Global state for loaded model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    # For demonstration, we load a default GCN model if weights exist, otherwise None
    model_type = os.getenv("MODEL_TYPE", "gcn")
    weights_path = os.getenv("MODEL_WEIGHTS", "best_model_gcn_bbbp.pt")
    
    if os.path.exists(weights_path):
        # We need to know in_dim. For our standard graphs, in_dim is usually 9.
        model = load_gnn_model(model_type=model_type, weights_path=weights_path, in_dim=9, hidden_dim=128)
        ml_models["model"] = model
        ml_models["explainer"] = get_explainer(model, task_type="binary_classification")
    else:
        ml_models["model"] = None
        ml_models["explainer"] = None
    yield
    # Cleanup on shutdown
    ml_models.clear()

app = FastAPI(title="Molecular Property Prediction API", lifespan=lifespan)

class PredictRequest(BaseModel):
    smiles: str
    explain: bool = False

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": ml_models.get("model") is not None}

@app.post("/predict")
async def predict(req: PredictRequest):
    if ml_models.get("model") is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    std_smiles = standardize_smiles(req.smiles)
    if not std_smiles:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")
        
    graph = smiles_to_graph(std_smiles)
    if not graph:
        raise HTTPException(status_code=400, detail="Could not extract features from SMILES.")
        
    # Standardize input for batch=1
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        
    # Inference
    with torch.no_grad():
        out = ml_models["model"](graph)
        pred = torch.sigmoid(out).item() 
        
    result = {
        "smiles": req.smiles,
        "standardized_smiles": std_smiles,
        "predictions": {
            "task_1": pred
        }
    }
    
    if req.explain and ml_models.get("explainer") is not None:
        explanation = explain_graph(
            ml_models["explainer"], 
            x=graph.x, 
            edge_index=graph.edge_index, 
            edge_attr=getattr(graph, 'edge_attr', None)
        )
        
        node_mask = explanation.node_mask
        if node_mask is not None:
            # sum over features for each node to get atom importance
            atom_importance = node_mask.sum(dim=1).tolist()
            result["explanation"] = {"atom_importance": atom_importance}
            
    return result
