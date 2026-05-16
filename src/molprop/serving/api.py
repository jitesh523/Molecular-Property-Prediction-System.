"""
Molecular Property Prediction API.

Production-grade FastAPI service for SMILES-to-prediction workflows
with batch support, model metadata, and explainability.
"""

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from threading import Lock
from typing import List, Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from molprop.data.smiles_vocab import SmilesVocab
from molprop.data.splits import generate_scaffold
from molprop.data.standardize import (
    ghose_filter,
    passes_lipinski_ro5,
    standardize_smiles,
    veber_filter,
)
from molprop.features.admet import compute_admet
from molprop.features.conformers import generate_3d_conformer, mol_to_pdb
from molprop.features.descriptors import get_descriptor_names, smiles_to_descriptors
from molprop.features.fingerprints import smiles_to_maccs, tanimoto_similarity
from molprop.features.graphs import smiles_to_graph
from molprop.models.explain import explain_graph, get_explainer
from molprop.models.optimization import LatentOptimizer
from molprop.models.pareto import ParetoOptimizer
from molprop.models.vae import SMILESVAE
from molprop.models.visualize_explanations import get_explanation_image
from molprop.serving.load_model import load_gnn_model
from molprop.serving.vector_db import vector_store
from molprop.storage.library import library

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Global state for loaded model + VAE
ml_models = {}
vae_state: dict = {}


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

        # --- Populate Vector DB ---
        try:
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            data_path = root_dir / "data" / "processed" / dataset / "processed.csv"
            if data_path.exists():
                df = pd.read_csv(data_path)
                log.info(f"Indexing {len(df)} molecules for vector search...")

                points = []
                for i, row in df.iterrows():
                    smiles = row["std_smiles"]
                    target = row.get(ml_models["task"], row.get(df.columns[-1]))
                    g = smiles_to_graph(smiles)
                    if g:
                        embedding = model.encode(g).squeeze(0).cpu().numpy().tolist()
                        if vector_store.vector_size is None:
                            vector_store.create_collection(len(embedding))
                        points.append(
                            {
                                "id": i,
                                "vector": embedding,
                                "payload": {
                                    "smiles": smiles,
                                    "task_value": float(target) if pd.notnull(target) else 0.0,
                                },
                            }
                        )
                vector_store.upsert_molecules(points)
                log.info("Vector DB indexing complete.")
        except Exception as e:
            log.error(f"Failed to populate vector DB: {e}")
    else:
        ml_models["model"] = None
        ml_models["explainer"] = None
        log.warning(f"No weights found at {weights_path}; API running without model.")

    # ── Load VAE (optional) ───────────────────────────────────────────────────
    vae_dataset = os.getenv("VAE_DATASET", dataset)
    vae_ckpt = ROOT / f"best_model_vae_{vae_dataset}.pt"
    vocab_path = ROOT / f"vocab_vae_{vae_dataset}.json"
    if vae_ckpt.exists() and vocab_path.exists():
        try:
            ck = torch.load(vae_ckpt, map_location="cpu", weights_only=False)  # nosec B614
            vocab = SmilesVocab.load(vocab_path)
            vae = SMILESVAE(
                vocab_size=ck["vocab_size"],
                latent_dim=ck["latent_dim"],
                hidden_dim=ck["hidden_dim"],
            )
            vae.load_state_dict(ck["state_dict"])
            vae.eval()
            vae_state["model"] = vae
            vae_state["vocab"] = vocab
            vae_state["max_len"] = ck.get("max_len", 120)
            log.info(f"VAE loaded from {vae_ckpt}")
        except Exception as exc:
            log.warning(f"VAE load failed: {exc}")
    else:
        log.info("No VAE checkpoint found; /generate endpoint inactive.")

    yield
    ml_models.clear()
    vae_state.clear()


# ── App Setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Molecular Property Prediction API",
    description=(
        "Production-grade REST API for predicting molecular properties from SMILES strings. "
        "Supports single and batch predictions, chemical standardization, GNNExplainer interpretability, "
        "KNN structural search, and generative molecular design via SMILES VAE."
    ),
    version="2.1.0",
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


# ── Lightweight in-process metrics registry ───────────────────────────────────

_metrics_lock = Lock()
_metrics: dict[str, dict] = {
    "requests_total": defaultdict(int),
    "requests_errors": defaultdict(int),
    "latency_sum_s": defaultdict(float),
    "started_at": time.time(),
}


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Process-Time header and record in-process metrics."""
    start = time.perf_counter()
    key = f"{request.method} {request.url.path}"
    try:
        response = await call_next(request)
    except Exception:
        with _metrics_lock:
            _metrics["requests_total"][key] += 1
            _metrics["requests_errors"][key] += 1
        raise
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
    with _metrics_lock:
        _metrics["requests_total"][key] += 1
        _metrics["latency_sum_s"][key] += elapsed
        if response.status_code >= 400:
            _metrics["requests_errors"][key] += 1
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
    similar_molecules: Optional[list[dict]] = None
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


@app.get("/version", tags=["System"])
async def version():
    """Return package version and key runtime metadata."""
    try:
        v = pkg_version("molprop")
    except PackageNotFoundError:
        v = "unknown"
    return {
        "version": v,
        "api_version": app.version,
        "torch": torch.__version__,
        "model_loaded": ml_models.get("model") is not None,
        "vae_loaded": bool(vae_state.get("model")),
    }


@app.get("/metrics", tags=["System"])
async def metrics():
    """Return simple in-process request metrics (JSON)."""
    with _metrics_lock:
        totals = dict(_metrics["requests_total"])
        errors = dict(_metrics["requests_errors"])
        latency_sum = dict(_metrics["latency_sum_s"])
        started = _metrics["started_at"]

    per_route = {}
    for key, count in totals.items():
        per_route[key] = {
            "requests": count,
            "errors": errors.get(key, 0),
            "avg_latency_s": (latency_sum.get(key, 0.0) / count) if count else 0.0,
        }
    return {
        "uptime_s": round(time.time() - started, 3),
        "routes": per_route,
    }


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

    # --- Vector Search for Structural Analogs ---
    try:
        query_vec = ml_models["model"].encode(graph).squeeze(0).cpu().numpy().tolist()
        result.similar_molecules = vector_store.search_similar(query_vec, top_k=5)
    except Exception as e:
        log.warning(f"Similarity search failed: {e}")

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
            result.explanation = {"atom_importance": atom_importance, "svg": svg_data}

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

    if len(req.smiles_list) == 0:
        raise HTTPException(status_code=400, detail="smiles_list must contain at least one entry.")
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


# ── Generative Design Endpoint ────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    n: int = Field(5, ge=1, le=20, description="Number of molecules to generate")
    temperature: float = Field(
        0.8, ge=0.1, le=2.0, description="Sampling temperature (lower = more conservative)"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class GeneratedMolecule(BaseModel):
    smiles: str
    standardized_smiles: Optional[str] = None
    valid: bool
    index: int


@app.post("/generate", response_model=list[GeneratedMolecule], tags=["Generative Design"])
async def generate_molecules(req: GenerateRequest):
    """
    Generate novel SMILES strings by sampling from the VAE latent space.

    Samples `n` latent vectors from N(0, I), decodes them to SMILES token
    sequences, and standardises the output with RDKit.
    """
    if not vae_state.get("model"):
        raise HTTPException(
            status_code=503,
            detail="VAE model not loaded. Train with `python scripts/train_vae.py` first.",
        )

    vae: SMILESVAE = vae_state["model"]
    vocab: SmilesVocab = vae_state["vocab"]
    max_len: int = vae_state["max_len"]

    if req.seed is not None:
        torch.manual_seed(req.seed)

    with torch.no_grad():
        z = torch.randn(req.n, vae.latent_dim)
        logits = vae.decode(z, max_len=max_len, temperature=req.temperature)
        token_ids = logits.argmax(dim=-1).cpu().tolist()

    results = []
    for i, ids in enumerate(token_ids):
        raw_smi = vocab.decode(ids)
        std_smi = standardize_smiles(raw_smi) if raw_smi else None
        results.append(
            GeneratedMolecule(
                smiles=raw_smi,
                standardized_smiles=std_smi,
                valid=bool(std_smi),
                index=i,
            )
        )

    return results


class SmartGenerateRequest(BaseModel):
    n_target: int = Field(20, ge=1, le=100, description="Number of valid passing molecules wanted")
    max_attempts: int = Field(
        500, ge=50, le=2000, description="Max sampling attempts before giving up"
    )
    temperature: float = Field(0.9, ge=0.1, le=2.0)
    targets: dict[str, tuple[float, float]] = Field(
        ...,
        description="Property -> (min, max) ranges. Supported: mw, logp, tpsa, hbd, hba, qed",
    )


@app.post("/generate/smart", tags=["Generative Design"])
async def generate_smart(req: SmartGenerateRequest):
    """
    Smart generation: sample molecules from the VAE and keep only those
    whose properties fall within all target ranges.

    Returns up to `n_target` valid molecules with their properties.
    """
    if not vae_state.get("model"):
        raise HTTPException(status_code=503, detail="VAE model not loaded.")

    from molprop.models.optimization import compute_qed, smiles_to_properties

    vae: SMILESVAE = vae_state["model"]
    vocab: SmilesVocab = vae_state["vocab"]
    max_len: int = vae_state["max_len"]

    valid_targets = {"mw", "logp", "tpsa", "hbd", "hba", "qed"}
    invalid = [t for t in req.targets if t not in valid_targets]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported targets: {invalid}. Supported: {sorted(valid_targets)}",
        )

    accepted = []
    attempts = 0
    batch_size = 50

    with torch.no_grad():
        while len(accepted) < req.n_target and attempts < req.max_attempts:
            z = torch.randn(batch_size, vae.latent_dim)
            logits = vae.decode(z, max_len=max_len, temperature=req.temperature)
            token_ids = logits.argmax(dim=-1).cpu().tolist()

            for ids in token_ids:
                attempts += 1
                if len(accepted) >= req.n_target or attempts >= req.max_attempts:
                    break
                raw_smi = vocab.decode(ids)
                std_smi = standardize_smiles(raw_smi) if raw_smi else None
                if not std_smi:
                    continue

                props = smiles_to_properties(std_smi)
                if not props:
                    continue
                # Add QED if needed
                if "qed" in req.targets:
                    qed_val = compute_qed(std_smi)
                    if qed_val is None:
                        continue
                    props["qed"] = qed_val

                # Check all target ranges
                passes = True
                for prop, (lo, hi) in req.targets.items():
                    val = props.get(prop)
                    if val is None or val < lo or val > hi:
                        passes = False
                        break

                if passes:
                    accepted.append(
                        {
                            "smiles": std_smi,
                            "properties": {
                                k: round(float(v), 3)
                                for k, v in props.items()
                                if isinstance(v, (int, float))
                            },
                        }
                    )

    return {
        "molecules": accepted,
        "n_accepted": len(accepted),
        "n_attempts": attempts,
        "acceptance_rate": round(len(accepted) / max(1, attempts), 4),
        "targets": {k: list(v) for k, v in req.targets.items()},
    }


@app.get("/generate/status", tags=["Generative Design"])
async def generate_status():
    """Check whether the VAE generator is loaded and ready."""
    return {
        "vae_loaded": bool(vae_state.get("model")),
        "latent_dim": vae_state["model"].latent_dim if vae_state.get("model") else None,
        "vocab_size": len(vae_state["vocab"]) if vae_state.get("vocab") else None,
    }


# ── Guided Molecular Optimization Endpoint ────────────────────────────────────


class OptimizeRequest(BaseModel):
    targets: dict[str, tuple[float, float]] = Field(
        ...,
        description="Property name -> (min, max) target ranges. Supported: mw, logp, tpsa, hbd, hba, qed, sas",
    )
    method: str = Field(
        "gradient_ascent", description="Optimization method: gradient_ascent or random_walk"
    )
    n_candidates: int = Field(
        10, ge=1, le=50, description="Number of candidate molecules to generate"
    )
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="VAE sampling temperature")
    seed_smiles: str | None = Field(
        None,
        max_length=MAX_SMILES_LEN,
        description="Optional seed molecule to start optimization from",
    )


class OptimizedMolecule(BaseModel):
    smiles: str
    score: float
    properties: dict[str, float]
    latent_vector: list[float]


class OptimizeResponse(BaseModel):
    method: str
    targets: dict[str, tuple[float, float]]
    candidates: list[OptimizedMolecule]
    total_attempts: int
    valid_count: int


@app.post("/optimize", response_model=OptimizeResponse, tags=["Generative Design"])
async def optimize_molecules(req: OptimizeRequest):
    """
    Guided molecular optimization toward target property constraints.

    Uses the trained VAE to navigate latent space and find molecules
    matching desired property ranges (MW, LogP, TPSA, HBD, HBA, QED, SAS).

    Supports starting from a seed molecule to explore its chemical neighborhood.

    Two methods available:
    - gradient_ascent: Uses finite-difference gradients to optimize latent vectors
    - random_walk: Samples and selects best matches (baseline)
    """
    if not vae_state.get("model"):
        raise HTTPException(
            status_code=503,
            detail="VAE model not loaded. Train with `python scripts/train_vae.py` first.",
        )

    vae = vae_state["model"]
    vocab = vae_state["vocab"]

    optimizer = LatentOptimizer(
        vae=vae,
        vocab=vocab,
        max_len=vae_state.get("max_len", 120),
        device="cpu",
    )

    try:
        results = optimizer.optimize(
            targets=req.targets,
            method=req.method,
            n_candidates=req.n_candidates,
            temperature=req.temperature,
            seed_smiles=req.seed_smiles,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc

    candidates = [
        OptimizedMolecule(
            smiles=r["smiles"],
            score=r["score"],
            properties=r["properties"],
            latent_vector=r["z"],
        )
        for r in results
        if r["smiles"]
    ]

    return OptimizeResponse(
        method=req.method,
        targets=req.targets,
        candidates=candidates,
        total_attempts=req.n_candidates * (50 if req.method == "gradient_ascent" else 10),
        valid_count=len(candidates),
    )


# ── Pareto Optimization Endpoint ──────────────────────────────────────────────


class ParetoRequest(BaseModel):
    objectives: list[str] = Field(
        ["qed", "neg_sas"],
        description=(
            "List of objectives to maximize simultaneously. "
            "Supported: qed, neg_sas, logp_norm, mw_norm, tpsa_norm, hbd_norm, hba_norm"
        ),
    )
    n_samples: int = Field(
        200, ge=20, le=1000, description="Number of molecules to sample before Pareto filtering"
    )
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="VAE sampling temperature")
    seed_smiles: str | None = Field(
        None, max_length=MAX_SMILES_LEN, description="Optional seed molecule"
    )


class ParetoMolecule(BaseModel):
    smiles: str
    objectives: dict[str, float]
    properties: dict[str, float]
    crowding_distance: float
    latent_vector: list[float]


class ParetoResponse(BaseModel):
    objectives: list[str]
    pareto_front: list[ParetoMolecule]
    total_sampled: int
    pareto_count: int
    dominated_count: int


@app.post("/optimize/pareto", response_model=ParetoResponse, tags=["Generative Design"])
async def optimize_pareto(req: ParetoRequest):
    """
    Multi-objective Pareto optimization in VAE latent space.

    Samples molecules and returns those on the Pareto front — optimal across
    all specified objectives with no trade-off possible.

    Objectives (all maximized):
    - qed: Quantitative Estimate of Drug-likeness (0-1)
    - neg_sas: Negative synthetic accessibility (easier synthesis = higher)
    - logp_norm: LogP proximity to 0-5 optimal range
    - mw_norm: MW proximity to 200-500 drug-like range
    - tpsa_norm: TPSA proximity to 20-140 optimal range
    - hbd_norm, hba_norm: H-bond counts within Lipinski range
    """
    if not vae_state.get("model"):
        raise HTTPException(
            status_code=503,
            detail="VAE model not loaded. Train with `python scripts/train_vae.py` first.",
        )

    vae = vae_state["model"]
    vocab = vae_state["vocab"]

    lat_opt = LatentOptimizer(
        vae=vae,
        vocab=vocab,
        max_len=vae_state.get("max_len", 120),
        device="cpu",
    )
    pareto_opt = ParetoOptimizer(lat_opt)

    try:
        result = pareto_opt.optimize_pareto(
            objectives=req.objectives,
            n_samples=req.n_samples,
            temperature=req.temperature,
            seed_smiles=req.seed_smiles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pareto optimization failed: {exc}") from exc

    pareto_molecules = [
        ParetoMolecule(
            smiles=m["smiles"],
            objectives=m["objectives"],
            properties=m["properties"],
            crowding_distance=m.get("crowding_distance", 0.0),
            latent_vector=m["z"],
        )
        for m in result["pareto_front"]
    ]

    return ParetoResponse(
        objectives=result["objectives"],
        pareto_front=pareto_molecules,
        total_sampled=result["total_sampled"],
        pareto_count=result["pareto_count"],
        dominated_count=result["dominated_count"],
    )


# ── Latent Space Map Endpoint ─────────────────────────────────────────────────


class LatentMapRequest(BaseModel):
    n_samples: int = Field(
        300, ge=50, le=2000, description="Number of molecules to sample for the map"
    )
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="VAE sampling temperature")
    seed_smiles: str | None = Field(
        None, max_length=MAX_SMILES_LEN, description="Optional seed molecule to highlight"
    )


class LatentMapPoint(BaseModel):
    x: float
    y: float
    smiles: str
    qed: float
    mw: float
    is_seed: bool = False


@app.post("/latent_map", tags=["Generative Design"])
async def latent_map(req: LatentMapRequest):
    """
    Generate a 2D projection of the VAE latent space using PCA.

    Samples molecules, computes their latent vectors, projects to 2D,
    and returns coordinates colored by QED drug-likeness.
    Optional seed molecule is highlighted in the projection.
    """
    if not vae_state.get("model"):
        raise HTTPException(
            status_code=503,
            detail="VAE model not loaded. Train with `python scripts/train_vae.py` first.",
        )

    from molprop.models.optimization import compute_qed, smiles_to_properties

    vae = vae_state["model"]
    vocab = vae_state["vocab"]
    max_len = vae_state.get("max_len", 120)

    lat_opt = LatentOptimizer(vae=vae, vocab=vocab, max_len=max_len, device="cpu")

    # Sample latent vectors
    batch_size = 50
    all_z = []
    all_smiles = []

    with torch.no_grad():
        for _ in range(req.n_samples // batch_size):
            z = torch.randn(batch_size, vae.latent_dim)
            smiles_list = lat_opt.decode_smiles(z, temperature=req.temperature)
            for smi, vec in zip(smiles_list, z, strict=False):
                if smi:
                    all_z.append(vec.cpu().numpy())
                    all_smiles.append(smi)

    # Optionally encode the seed
    seed_z_np = None
    seed_smi = None
    if req.seed_smiles:
        seed_tensor = lat_opt.encode_seed(req.seed_smiles)
        if seed_tensor is not None:
            seed_z_np = seed_tensor.squeeze(0).cpu().numpy()
            seed_smi = req.seed_smiles
            all_z.append(seed_z_np)
            all_smiles.append(seed_smi)

    if len(all_z) < 5:
        raise HTTPException(
            status_code=500, detail="Could not decode enough valid molecules for the map."
        )

    import numpy as np

    Z = np.array(all_z)

    # PCA to 2D (no external dependency — always available)
    Z_centered = Z - Z.mean(axis=0)
    cov = np.cov(Z_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top-2 eigenvectors (largest eigenvalues)
    idx = np.argsort(eigenvalues)[::-1]
    pcs = eigenvectors[:, idx[:2]]
    Z_2d = Z_centered @ pcs

    points = []
    for _i, (smi, coords) in enumerate(zip(all_smiles, Z_2d, strict=False)):
        props = smiles_to_properties(smi) or {}
        qed_val = compute_qed(smi) or 0.0
        is_seed = seed_smi is not None and smi == seed_smi
        points.append(
            LatentMapPoint(
                x=float(coords[0]),
                y=float(coords[1]),
                smiles=smi,
                qed=round(qed_val, 4),
                mw=round(float(props.get("mw", 0.0)), 2),
                is_seed=is_seed,
            )
        )

    return {
        "points": [p.model_dump() for p in points],
        "total": len(points),
        "projection": "PCA",
        "color_by": "qed",
    }


# ── Descriptors Endpoint ──────────────────────────────────────────────────────


class DescriptorRequest(BaseModel):
    smiles: str = Field(..., max_length=MAX_SMILES_LEN, description="SMILES string")
    include_fingerprint: bool = Field(
        False, description="Include 167-bit MACCS keys fingerprint in response"
    )


class DescriptorResponse(BaseModel):
    smiles: str
    standardized_smiles: Optional[str] = None
    descriptors: Optional[dict[str, float]] = None
    maccs_fingerprint: Optional[list[int]] = None
    error: Optional[str] = None


@app.post("/descriptors", response_model=DescriptorResponse, tags=["Cheminformatics"])
async def compute_descriptors(req: DescriptorRequest):
    """
    Compute physicochemical descriptors (and optionally MACCS keys) for a SMILES.

    Returns 18 ADMET/Lipinski-relevant descriptors including LogP, MW, TPSA,
    H-bond donors/acceptors, ring counts, FractionCSP3, BertzCT, and more.
    """
    std_smiles = standardize_smiles(req.smiles)
    if not std_smiles:
        return DescriptorResponse(smiles=req.smiles, error="Invalid SMILES string.")

    desc_arr = smiles_to_descriptors(std_smiles)
    if desc_arr is None:
        return DescriptorResponse(
            smiles=req.smiles,
            standardized_smiles=std_smiles,
            error="Failed to compute descriptors.",
        )

    desc_dict = dict(zip(get_descriptor_names(), desc_arr.tolist(), strict=False))

    maccs = None
    if req.include_fingerprint:
        fp = smiles_to_maccs(std_smiles)
        maccs = fp.tolist() if fp is not None else None

    return DescriptorResponse(
        smiles=req.smiles,
        standardized_smiles=std_smiles,
        descriptors=desc_dict,
        maccs_fingerprint=maccs,
    )


# ── ADMET Property Prediction Endpoint ────────────────────────────────────────


class ADMETRequest(BaseModel):
    smiles: str = Field(..., max_length=MAX_SMILES_LEN, description="SMILES string")


@app.post("/admet", tags=["Cheminformatics"])
async def admet_prediction(req: ADMETRequest):
    """
    Compute ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties.

    Uses RDKit rule-based filters, PAINS/Brenk structural alerts, and
    established medicinal chemistry rules to assess drug-likeness safety.

    Returns:
    - Absorption: Lipinski Ro5, TPSA, oral bioavailability
    - Distribution: BBB permeability, fraction Csp3
    - Metabolism: CYP450 inhibition risk flags
    - Excretion: Renal clearance estimate
    - Toxicity: hERG risk, Ames mutagenicity, PAINS/Brenk alerts
    - overall_score: 0-100 composite score (≥60 = pass)
    """
    result = compute_admet(req.smiles)
    if result is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: '{req.smiles}'")
    return result.to_dict()


class ADMETBatchRequest(BaseModel):
    smiles_list: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


@app.post("/admet/batch", tags=["Cheminformatics"])
async def admet_batch(req: ADMETBatchRequest):
    """Compute ADMET for a batch of SMILES strings."""
    results = []
    for smi in req.smiles_list:
        r = compute_admet(smi)
        results.append(r.to_dict() if r else {"smiles": smi, "error": "Invalid SMILES"})
    return {"results": results, "count": len(results)}


# ── Batch Descriptor Endpoint ───────────────────────────────────────────────


class BatchDescriptorRequest(BaseModel):
    smiles_list: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="List of SMILES strings (max 100)",
    )


class BatchDescriptorItem(BaseModel):
    smiles: str
    standardized_smiles: Optional[str] = None
    descriptors: Optional[dict[str, float]] = None
    error: Optional[str] = None


class BatchDescriptorResponse(BaseModel):
    results: List[BatchDescriptorItem]
    total: int
    failed: int


@app.post("/batch/descriptors", response_model=BatchDescriptorResponse, tags=["Cheminformatics"])
async def batch_descriptors(req: BatchDescriptorRequest):
    """
    Compute physicochemical descriptors for a batch of SMILES strings.

    Accepts up to ``MAX_BATCH_SIZE`` (100) molecules per call. Each result
    contains the same 18 ADMET/Lipinski descriptors as the single-molecule
    ``/descriptors`` endpoint, plus an error field for invalid entries.
    Summary counts (total, failed) are included at the top level.
    """
    results: List[BatchDescriptorItem] = []
    failed = 0
    for smiles in req.smiles_list:
        std = standardize_smiles(smiles)
        if not std:
            results.append(BatchDescriptorItem(smiles=smiles, error="Invalid SMILES."))
            failed += 1
            continue
        desc_arr = smiles_to_descriptors(std)
        if desc_arr is None:
            results.append(
                BatchDescriptorItem(
                    smiles=smiles,
                    standardized_smiles=std,
                    error="Descriptor computation failed.",
                )
            )
            failed += 1
            continue
        desc_dict = dict(zip(get_descriptor_names(), desc_arr.tolist(), strict=False))
        results.append(
            BatchDescriptorItem(smiles=smiles, standardized_smiles=std, descriptors=desc_dict)
        )
    return BatchDescriptorResponse(results=results, total=len(results), failed=failed)


# ── Scaffold Endpoint ─────────────────────────────────────────────────────────


@app.get("/scaffold", tags=["Cheminformatics"])
async def get_scaffold(smiles: str):
    """
    Return the Bemis–Murcko scaffold for a SMILES string.

    Scaffolds represent the ring systems and linkers of a molecule after
    side-chain stripping. They are the standard unit for scaffold hopping,
    chemical diversity analysis, and dataset stratification.
    Returns ``null`` for molecules with no ring systems (acyclic compounds).
    """
    std = standardize_smiles(smiles)
    if not std:
        raise HTTPException(status_code=400, detail="Invalid or unparseable SMILES.")
    scaffold = generate_scaffold(std) or None
    return {"smiles": smiles, "standardized_smiles": std, "scaffold": scaffold}


# ── Similarity Search Endpoint ───────────────────────────────────────────────


@app.get("/search", tags=["Cheminformatics"])
async def similarity_search(smiles: str, top_k: int = 5):
    """
    Find the most structurally similar molecules in the indexed database.

    Encodes the query SMILES as a GNN embedding vector and performs cosine
    KNN search against the Qdrant vector store.
    Returns up to `top_k` neighbours with their SMILES, score, and task value.
    """
    if ml_models.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded; vector search unavailable.")
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50.")

    std_smiles = standardize_smiles(smiles)
    if not std_smiles:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    graph = smiles_to_graph(std_smiles)
    if graph is None:
        raise HTTPException(status_code=400, detail="Could not extract features from SMILES.")

    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    try:
        query_vec = ml_models["model"].encode(graph).squeeze(0).cpu().numpy().tolist()
        results = vector_store.search_similar(query_vec, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {exc}") from exc

    return {
        "query_smiles": smiles,
        "standardized_smiles": std_smiles,
        "indexed_molecules": vector_store.count(),
        "results": results,
    }


# ── Fingerprint Similarity Search ─────────────────────────────────────────────


class SimilaritySearchRequest(BaseModel):
    smiles: str = Field(..., max_length=MAX_SMILES_LEN, description="Query SMILES string")
    top_k: int = Field(10, ge=1, le=50, description="Number of similar molecules to return")
    fingerprint_type: str = Field(
        "morgan", pattern="^(morgan|maccs)$", description="Fingerprint type: morgan or maccs"
    )
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")


class SimilarityResult(BaseModel):
    smiles: str
    score: float
    rank: int


@app.post("/search/similar", tags=["Search"])
async def fingerprint_similarity_search(req: SimilaritySearchRequest):
    """
    Find chemically similar molecules using fingerprint-based similarity.

    Uses either Morgan (circular, ECFP-like) or MACCS fingerprints with
    Tanimoto coefficient. Searches against the indexed molecule database.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: '{req.smiles}'")

    # Generate query fingerprint
    if req.fingerprint_type == "morgan":
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    else:
        from rdkit.Chem import MACCSkeys

        query_fp = MACCSkeys.GenMACCSKeys(mol)

    # Search in vector DB if populated (fallback: empty)
    if vector_store.count() == 0:
        raise HTTPException(
            status_code=503, detail="Vector database is empty. Index molecules first."
        )

    # Get all indexed molecules and compute similarities
    # In production, this should use a fingerprint index (RDKit cartridge, etc.)
    # For now, we search through the indexed set
    all_results = []
    for point in vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )[0]:
        smi = point.payload.get("smiles", "")
        if not smi:
            continue
        target_mol = Chem.MolFromSmiles(smi)
        if target_mol is None:
            continue

        if req.fingerprint_type == "morgan":
            target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
        else:
            from rdkit.Chem import MACCSkeys

            target_fp = MACCSkeys.GenMACCSKeys(target_mol)

        sim = DataStructs.TanimotoSimilarity(query_fp, target_fp)
        if sim >= req.threshold:
            all_results.append({"smiles": smi, "score": sim, "payload": point.payload})

    # Sort by similarity descending
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = all_results[: req.top_k]

    return {
        "query_smiles": req.smiles,
        "fingerprint_type": req.fingerprint_type,
        "threshold": req.threshold,
        "total_indexed": vector_store.count(),
        "results": [
            SimilarityResult(
                smiles=r["smiles"], score=round(r["score"], 4), rank=i + 1
            ).model_dump()
            for i, r in enumerate(top_results)
        ],
    }


# ── Compound Library Endpoints ────────────────────────────────────────────────


class LibrarySaveRequest(BaseModel):
    smiles: str = Field(..., max_length=MAX_SMILES_LEN)
    name: Optional[str] = Field(None, max_length=200)
    project: str = Field("default", max_length=80)
    tags: list[str] = Field(default_factory=list)
    properties: dict = Field(default_factory=dict)
    notes: Optional[str] = Field(None, max_length=2000)


class LibraryUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    tags: Optional[list[str]] = None
    notes: Optional[str] = Field(None, max_length=2000)
    properties: Optional[dict] = None


@app.post("/library", tags=["Library"])
async def library_save(req: LibrarySaveRequest):
    """Save a compound to the library (idempotent on (smiles, project))."""
    std_smi = standardize_smiles(req.smiles)
    if not std_smi:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: '{req.smiles}'")
    return library.add(
        smiles=std_smi,
        name=req.name,
        project=req.project,
        tags=req.tags,
        properties=req.properties,
        notes=req.notes,
    )


@app.get("/library", tags=["Library"])
async def library_list(
    project: Optional[str] = None,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List compounds in the library, with optional filters."""
    return {
        "compounds": library.list(
            project=project, tag=tag, search=search, limit=limit, offset=offset
        ),
        "total": library.count(project=project),
    }


@app.get("/library/projects", tags=["Library"])
async def library_projects():
    """Return all known projects + tags."""
    return {"projects": library.projects(), "tags": library.all_tags()}


@app.get("/library/{compound_id}", tags=["Library"])
async def library_get(compound_id: int):
    """Get a single compound by ID."""
    result = library.get(compound_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Compound {compound_id} not found")
    return result


@app.patch("/library/{compound_id}", tags=["Library"])
async def library_update(compound_id: int, req: LibraryUpdateRequest):
    """Update a compound's metadata."""
    result = library.update(
        compound_id,
        name=req.name,
        tags=req.tags,
        notes=req.notes,
        properties=req.properties,
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Compound {compound_id} not found")
    return result


@app.delete("/library/{compound_id}", tags=["Library"])
async def library_delete(compound_id: int):
    """Delete a compound by ID."""
    if not library.delete(compound_id):
        raise HTTPException(status_code=404, detail=f"Compound {compound_id} not found")
    return {"status": "deleted", "id": compound_id}


# ── Conformer Endpoint ────────────────────────────────────────────────────────


class ConformerRequest(BaseModel):
    smiles: str = Field(..., max_length=MAX_SMILES_LEN, description="SMILES string")


class ConformerResponse(BaseModel):
    smiles: str
    standardized_smiles: Optional[str] = None
    pdb_block: Optional[str] = None
    num_atoms: Optional[int] = None
    error: Optional[str] = None


@app.post("/conformer", response_model=ConformerResponse, tags=["Cheminformatics"])
async def get_conformer(req: ConformerRequest):
    """
    Generate a 3D conformer for a SMILES string.

    Uses RDKit ETKDGv3 embedding followed by MMFF94 force-field optimisation.
    Returns the optimized geometry as a PDB block for downstream 3D visualisation
    (e.g. with py3Dmol / 3Dmol.js).
    """
    std_smiles = standardize_smiles(req.smiles)
    if not std_smiles:
        return ConformerResponse(smiles=req.smiles, error="Invalid SMILES string.")

    mol = generate_3d_conformer(std_smiles)
    if mol is None:
        return ConformerResponse(
            smiles=req.smiles,
            standardized_smiles=std_smiles,
            error="3D embedding failed for this molecule.",
        )

    pdb = mol_to_pdb(mol)
    return ConformerResponse(
        smiles=req.smiles,
        standardized_smiles=std_smiles,
        pdb_block=pdb,
        num_atoms=mol.GetNumAtoms(),
    )


# ── Lipinski Rule-of-Five Endpoint ────────────────────────────────────────────


class LipinskiResponse(BaseModel):
    smiles: str
    standardized_smiles: Optional[str] = None
    passes: Optional[bool] = None
    violations: Optional[list[str]] = None
    MW: Optional[float] = None
    LogP: Optional[float] = None
    HBD: Optional[int] = None
    HBA: Optional[int] = None
    error: Optional[str] = None


@app.get("/lipinski", response_model=LipinskiResponse, tags=["Cheminformatics"])
async def check_lipinski(smiles: str):
    """
    Evaluate Lipinski's Rule of Five for a SMILES string.

    Returns individual property values (MW, LogP, HBD, HBA) and a list of
    violations. A molecule *passes* if it has at most 1 violation.
    """
    result = passes_lipinski_ro5(smiles)
    if result is None:
        return LipinskiResponse(smiles=smiles, error="Invalid or unparseable SMILES.")

    std = standardize_smiles(smiles)
    return LipinskiResponse(
        smiles=smiles,
        standardized_smiles=std,
        passes=result["passes"],
        violations=result["violations"],
        MW=result["MW"],
        LogP=result["LogP"],
        HBD=result["HBD"],
        HBA=result["HBA"],
    )


# ── Drug-Likeness Panel Endpoint ──────────────────────────────────────────────


class DrugLikenessResponse(BaseModel):
    smiles: str
    standardized_smiles: Optional[str] = None
    lipinski: Optional[dict] = None
    veber: Optional[dict] = None
    ghose: Optional[dict] = None
    overall_drug_like: Optional[bool] = None
    error: Optional[str] = None


@app.get("/druglikeness", response_model=DrugLikenessResponse, tags=["Cheminformatics"])
async def check_druglikeness(smiles: str):
    """
    Comprehensive drug-likeness panel for a SMILES string.

    Evaluates three complementary filters in a single call:

    - **Lipinski Ro5** — oral bioavailability heuristic (MW ≤ 500, LogP ≤ 5,
      HBD ≤ 5, HBA ≤ 10; max 1 violation allowed).
    - **Veber** — oral bioavailability via conformational flexibility
      (RotatableBonds ≤ 10 **and** TPSA ≤ 140 Å²; both must pass).
    - **Ghose** — global drug-like property ranges (−0.4 ≤ LogP ≤ 5.6,
      160 ≤ MW ≤ 480, 40 ≤ MR ≤ 130, 20 ≤ NumAtoms ≤ 70).

    `overall_drug_like` is `true` only when **all three** filters pass.
    """
    std = standardize_smiles(smiles)
    if not std:
        return DrugLikenessResponse(smiles=smiles, error="Invalid or unparseable SMILES.")

    lip = passes_lipinski_ro5(std)
    veb = veber_filter(std)
    gho = ghose_filter(std)

    overall = None
    if lip is not None and veb is not None and gho is not None:
        overall = lip["passes"] and veb["passes"] and gho["passes"]

    return DrugLikenessResponse(
        smiles=smiles,
        standardized_smiles=std,
        lipinski=lip,
        veber=veb,
        ghose=gho,
        overall_drug_like=overall,
    )


# ── Molecule Comparison Endpoint ──────────────────────────────────────────────


class CompareRequest(BaseModel):
    smiles_a: str = Field(..., max_length=MAX_SMILES_LEN, description="First SMILES string")
    smiles_b: str = Field(..., max_length=MAX_SMILES_LEN, description="Second SMILES string")


class MoleculeProfile(BaseModel):
    smiles: str
    standardized_smiles: Optional[str] = None
    descriptors: Optional[dict[str, float]] = None
    lipinski: Optional[dict] = None
    error: Optional[str] = None


class CompareResponse(BaseModel):
    molecule_a: MoleculeProfile
    molecule_b: MoleculeProfile
    tanimoto_similarity: Optional[float] = None


def _build_profile(smiles: str) -> MoleculeProfile:
    """Compute standardized SMILES, descriptors and Lipinski Ro5 for one molecule."""
    std = standardize_smiles(smiles)
    if not std:
        return MoleculeProfile(smiles=smiles, error="Invalid or unparseable SMILES.")
    desc_arr = smiles_to_descriptors(std)
    desc_dict = (
        dict(zip(get_descriptor_names(), desc_arr.tolist(), strict=False))
        if desc_arr is not None
        else None
    )
    ro5 = passes_lipinski_ro5(std)
    return MoleculeProfile(
        smiles=smiles,
        standardized_smiles=std,
        descriptors=desc_dict,
        lipinski=ro5,
    )


@app.post("/compare", response_model=CompareResponse, tags=["Cheminformatics"])
async def compare_molecules(req: CompareRequest):
    """
    Compare two molecules side-by-side.

    Returns physicochemical descriptors, Lipinski Ro5 evaluation, and Morgan
    fingerprint-based Tanimoto similarity (ECFP4, 2048 bits) for both SMILES.
    Tanimoto similarity of 1.0 indicates identical structures; 0.0 indicates no
    shared substructures in the fingerprint.
    """
    profile_a = _build_profile(req.smiles_a)
    profile_b = _build_profile(req.smiles_b)

    sim = None
    if profile_a.standardized_smiles and profile_b.standardized_smiles:
        sim = tanimoto_similarity(profile_a.standardized_smiles, profile_b.standardized_smiles)

    return CompareResponse(molecule_a=profile_a, molecule_b=profile_b, tanimoto_similarity=sim)


# Mount static files at the root (ensure this is after all other routes)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
