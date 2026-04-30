"""
Molecular Property Prediction API.

Production-grade FastAPI service for SMILES-to-prediction workflows
with batch support, model metadata, and explainability.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from molprop.data.smiles_vocab import SmilesVocab
from molprop.data.standardize import passes_lipinski_ro5, standardize_smiles
from molprop.features.conformers import generate_3d_conformer, mol_to_pdb
from molprop.features.descriptors import get_descriptor_names, smiles_to_descriptors
from molprop.features.fingerprints import smiles_to_maccs
from molprop.features.graphs import smiles_to_graph
from molprop.models.explain import explain_graph, get_explainer
from molprop.models.vae import SMILESVAE
from molprop.models.visualize_explanations import get_explanation_image
from molprop.serving.load_model import load_gnn_model
from molprop.serving.vector_db import vector_store

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
            ck = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
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
    version="2.0.0",
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


@app.get("/generate/status", tags=["Generative Design"])
async def generate_status():
    """Check whether the VAE generator is loaded and ready."""
    return {
        "vae_loaded": bool(vae_state.get("model")),
        "latent_dim": vae_state["model"].latent_dim if vae_state.get("model") else None,
        "vocab_size": len(vae_state["vocab"]) if vae_state.get("vocab") else None,
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

    desc_dict = dict(zip(get_descriptor_names(), desc_arr.tolist()))

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


# Mount static files at the root (ensure this is after all other routes)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
