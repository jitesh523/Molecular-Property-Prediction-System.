"""
Molecular Property Prediction API.

Production-grade FastAPI service for SMILES-to-prediction workflows
with batch support, model metadata, and explainability.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from molprop.data.standardize import standardize_smiles
from molprop.features.graphs import smiles_to_graph
from molprop.models.explain import explain_graph, get_explainer
from molprop.models.visualize_explanations import get_explanation_image
from molprop.serving.load_model import load_gnn_model

log = logging.getLogger(__name__)

# Global state for loaded model
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    model_type = os.getenv("MODEL_TYPE", "gcn")
    weights_path = os.getenv("MODEL_WEIGHTS", "best_model_gcn_bbbp.pt")
    dataset = os.getenv("MODEL_DATASET", "bbbp")
    task = os.getenv("MODEL_TASK", "classification")

    if os.path.exists(weights_path):
        model = load_gnn_model(
            model_type=model_type, weights_path=weights_path, in_dim=9, hidden_dim=128
        )
        ml_models["model"] = model
        ml_models["model_type"] = model_type
        ml_models["weights_path"] = weights_path
        ml_models["dataset"] = dataset
        ml_models["task"] = task
        ml_models["explainer"] = get_explainer(model, task_type="binary_classification")
        log.info(f"Model loaded: {model_type} ({weights_path})")
    else:
        ml_models["model"] = None
        ml_models["explainer"] = None
        log.warning(f"No weights found at {weights_path}; API running without model.")
    yield
    ml_models.clear()


# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Molecular Property Prediction API",
    description=(
        "Production-grade REST API for predicting molecular properties from SMILES strings. "
        "Supports single and batch predictions, chemical standardization, and "
        "GNNExplainer-based atom-level interpretability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for interactive demos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Timing Middleware ─────────────────────────────────────────────────────────


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Process-Time header for observability."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
    return response


# ── Request / Response Models ─────────────────────────────────────────────────

MAX_BATCH_SIZE = 100
MAX_SMILES_LEN = 500


class PredictRequest(BaseModel):
    smiles: str = Field(..., max_length=MAX_SMILES_LEN, description="SMILES string")
    explain: bool = Field(False, description="Include atom-level importance explanation")
    uncertainty_samples: int = Field(
        0, ge=0, le=50, description="Number of MC Dropout samples for uncertainty estimation"
    )


class BatchPredictRequest(BaseModel):
    smiles_list: list[str] = Field(
        ...,
        max_length=MAX_BATCH_SIZE,
        description=f"List of SMILES (max {MAX_BATCH_SIZE})",
    )
    explain: bool = Field(False, description="Include explanations for each molecule")
    uncertainty_samples: int = Field(
        0, ge=0, le=50, description="Number of MC Dropout samples for uncertainty estimation"
    )


class PredictionResult(BaseModel):
    smiles: str
    standardized_smiles: str
    predictions: dict[str, float]
    uncertainty_std: Optional[dict[str, float]] = None
    explanation: Optional[dict] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    model_type: str
    dataset: str
    task: str
    weights_path: str
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": ml_models.get("model") is not None}


@app.get("/model/info", response_model=ModelInfo, tags=["System"])
async def model_info():
    """Return metadata about the currently loaded model."""
    if ml_models.get("model") is None:
        return ModelInfo(
            model_type="none",
            dataset="none",
            task="none",
            weights_path="none",
            status="not_loaded",
        )
    return ModelInfo(
        model_type=ml_models.get("model_type", "unknown"),
        dataset=ml_models.get("dataset", "unknown"),
        task=ml_models.get("task", "unknown"),
        weights_path=ml_models.get("weights_path", "unknown"),
        status="loaded",
    )


def _predict_single(
    smiles: str, explain: bool = False, uncertainty_samples: int = 0
) -> PredictionResult:
    """Core prediction logic for a single SMILES."""
    std_smiles = standardize_smiles(smiles)
    if not std_smiles:
        return PredictionResult(
            smiles=smiles,
            standardized_smiles="",
            predictions={},
            error="Invalid SMILES string.",
        )

    graph = smiles_to_graph(std_smiles)
    if not graph:
        return PredictionResult(
            smiles=smiles,
            standardized_smiles=std_smiles,
            predictions={},
            error="Could not extract features from SMILES.",
        )

    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)

    with torch.no_grad():
        # Standard prediction
        out = ml_models["model"](graph)
        pred = torch.sigmoid(out).item()

        # Uncertainty estimation via MC Dropout
        uncertainty_std = None
        if uncertainty_samples > 0:
            samples = []
            for _ in range(uncertainty_samples):
                s_out = ml_models["model"](graph, mc_dropout=True)
                samples.append(torch.sigmoid(s_out).item())

            samples_ts = torch.tensor(samples)
            uncertainty_std = {"task_1": round(samples_ts.std().item(), 6)}

    result = PredictionResult(
        smiles=smiles,
        standardized_smiles=std_smiles,
        predictions={"task_1": round(pred, 6)},
        uncertainty_std=uncertainty_std,
    )

    if explain and ml_models.get("explainer") is not None:
        explanation = explain_graph(
            ml_models["explainer"],
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=getattr(graph, "edge_attr", None),
        )
        node_mask = explanation.node_mask
        if node_mask is not None:
            atom_importance = node_mask.sum(dim=1).tolist()
            svg_data = get_explanation_image(std_smiles, explanation)
            result.explanation = {
                "atom_importance": atom_importance,
                "svg": svg_data
            }

    return result


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict(req: PredictRequest):
    """Predict molecular properties for a single SMILES string."""
    if ml_models.get("model") is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

    result = _predict_single(req.smiles, req.explain, req.uncertainty_samples)
    if result.error:
        raise HTTPException(status_code=400, detail=result.error)
    return result


@app.post(
    "/predict/batch",
    response_model=list[PredictionResult],
    tags=["Prediction"],
)
async def predict_batch(req: BatchPredictRequest):
    """
    Predict molecular properties for a batch of SMILES strings.

    Returns a list of results, one per input SMILES. Individual failures
    are reported in the `error` field rather than failing the whole batch.
    """
    if ml_models.get("model") is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

    if len(req.smiles_list) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.smiles_list)} exceeds maximum of {MAX_BATCH_SIZE}.",
        )

    results = []
    for smiles in req.smiles_list:
        result = _predict_single(smiles, req.explain, req.uncertainty_samples)
        results.append(result)

    return results

# Mount static files at the root (ensure this is after all other routes)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

