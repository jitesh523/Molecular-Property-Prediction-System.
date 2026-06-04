# 🧬 Molecular Property Prediction System

A production-grade, end-to-end machine learning platform for predicting molecular properties from chemical structures. Built as a comprehensive ML engineering portfolio piece demonstrating the full lifecycle—from raw data ingestion to **3D structural analysis** and **interactive dashboard deployment**.

[![CI](https://github.com/jitesh523/Molecular-Property-Prediction-System./actions/workflows/ci.yml/badge.svg)](https://github.com/jitesh523/Molecular-Property-Prediction-System./actions)
[![Benchmark CI](https://github.com/jitesh523/Molecular-Property-Prediction-System./actions/workflows/benchmarks.yml/badge.svg)](https://github.com/jitesh523/Molecular-Property-Prediction-System./actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Interactive Demo](https://img.shields.io/badge/demo-streamlit-red.svg)](http://localhost:8501)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Portfolio Completion: 100%](https://img.shields.io/badge/portfolio-complete-brightgreen.svg)](docs/PORTFOLIO.md)

---

## � What's New in v2.x — Cheminformatics Suite

A full cheminformatics layer has been added on top of the original GNN platform.
Every feature is exposed via REST API, the Python SDK (`molprop.client`), and
the `molprop` command-line tool.

### REST endpoints (30+)

| Domain | Endpoints |
|---|---|
| **Prediction** | `POST /predict`, `POST /predict/batch` |
| **Generation** | `POST /generate`, `POST /generate/smart` (constrained by MW/LogP/QED/TPSA) |
| **ADMET** | `POST /admet`, `POST /admet/batch` |
| **Cheminformatics** | `POST /scaffold`, `POST /scaffold/batch`, `POST /functional_groups`, `POST /isomers`, `POST /substructure`, `POST /compare`, `POST /mcs`, `POST /alerts`, `POST /standardize`, `POST /conformer` |
| **Similarity** | `POST /search/similar` (Morgan / MACCS Tanimoto) |
| **Reports** | `POST /report` (one-click Markdown report) |
| **Library** | `POST /library`, `GET /library`, `GET /library/{id}`, `PATCH /library/{id}`, `DELETE /library/{id}`, `GET /library/projects`, `GET /library/export/csv`, `POST /library/import` |

### Premium UI — 14 tabs

`Predict` · `Generate` · `Optimize` · `History` · `Visualize` · `ADMET` ·
`Batch` · `Search` · `3D View` · `Dashboard` · `Library` · `Scaffold` ·
`Compare` · `Isomers`

Highlights:
- **Scaffold tab** — gradient SAScore meter (1 = easy → 10 = hard), ring topology cards, functional-group chips, **PAINS / Brenk / NIH structural alerts**.
- **Compare tab** — colour-coded MACCS Tanimoto headline, side-by-side drug-likeness panels, **Maximum Common Substructure (MCS)** with coverage %, descriptor delta table.
- **Isomers tab** — tautomer & stereoisomer enumeration with canonical-tautomer highlighting.
- **Library tab** — persistent SQLite-backed CRUD, projects, tags, full-text search, CSV import/export.

### Python SDK

```python
from molprop.client import MolpropClient
c = MolpropClient("http://localhost:8000")

c.predict("CC(=O)Oc1ccccc1C(=O)O")
c.scaffold("CC(=O)NC1=CC=C(O)C=C1")          # Bemis–Murcko + SAScore
c.mcs("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1")
c.alerts("Cc1ccc(cc1)/N=N/c2ccc(cc2)N(C)C")  # PAINS / Brenk / NIH
c.library_save("CCO", project="solvents", tags=["polar"])
c.substructure("c1ccccc1", project="solvents")
```

26 typed methods, uniform `MolpropAPIError` for non-2xx responses. See `docs/CLI_AND_SDK.md`.

### Command-line interface

```bash
molprop predict "CC(=O)Oc1ccccc1C(=O)O"
molprop scaffold "CC(=O)NC1=CC=C(O)C=C1"
molprop compare "CCO" "CCN"
molprop alerts "Cc1ccc(cc1)/N=N/c2ccc(cc2)N(C)C" --catalog PAINS --catalog BRENK
molprop report "aspirin_smiles_here" -o aspirin.md
molprop library save "CCO" --name Ethanol --project drug-x --tag solvent
molprop library list --project drug-x
```

`MOLPROP_URL` env-var overrides the API base. 19 subcommands total.

### Tests

- `tests/test_storage.py` — 9 tests (Library CRUD, upsert, persistence)
- `tests/test_cheminformatics_features.py` — 18 tests (scaffolds, FG, isomers, substructure)
- `tests/test_api_integration.py` — 14 end-to-end tests via FastAPI `TestClient`

```bash
pytest tests/test_storage.py tests/test_cheminformatics_features.py tests/test_api_integration.py -v
```

---

## �🏗️ Architecture

```mermaid
flowchart LR
    A["Raw Sources<br/>MoleculeNet · ChEMBL · PubChem"] --> B["Ingestion Layer"]
    B --> C["Chemical Standardization<br/>RDKit: salts, tautomers, canonical SMILES"]
    C --> D["Featurization"]
    D --> E["Morgan FP (2048-bit)"]
    D --> F["RDKit Descriptors"]
    D --> G["PyG Molecular Graphs"]
    E --> H["RF / XGBoost"]
    F --> H
    G --> I["GCN / GAT / MPNN"]
    H --> J["Evaluation<br/>Scaffold Split · RMSE · ROC-AUC · PR-AUC"]
    I --> J
    J --> K["Interpretability<br/>SHAP · GNNExplainer"]
    J --> L["Model Registry<br/>MLflow"]
    L --> M["FastAPI + Docker"]
```

---

## 📊 Datasets

| Dataset | Source | Task | Molecules | Endpoint |
|---------|--------|------|-----------|----------|
| ESOL (Delaney) | MoleculeNet | Regression | ~1,128 | Aqueous solubility (logS) |
| FreeSolv | MoleculeNet | Regression | ~643 | Hydration free energy |
| Lipophilicity | MoleculeNet | Regression | ~4,200 | Octanol/water partition (logD) |
| BBBP | MoleculeNet | Classification | ~2,039 | Blood-brain barrier permeability |
| ChEMBL EGFR | ChEMBL 36 | Regression | Variable | pIC50 inhibition potency |
| PubChem AID 260895 | PUG-REST | Regression | ~25 | erbB1 inhibition (IC50→pIC50) |

---

## 🖥️ Interactive Exploration Dashboards

The system includes two premium interactive web interfaces for real-time model interaction and chemical space visualization.

### 1. Premium Web UI (FastAPI Native)
A lightning-fast, glassmorphism-inspired HTML/Vanilla CSS frontend built for maximum performance and visual excellence.
- **Micro-animations & Modern Typography:** Sleek dark-mode interface with Inter font.
- **Instant Predictions:** Connects directly to the FastAPI backend without full-page reloads.
- **Structural Explanations:** Dynamic rendering of SVG evidence from `GNNExplainer`.

**Access:** Served directly from `http://localhost:8000/` when running the API.

### 2. Streamlit Analytical Dashboard
An analytical interface optimized for exploring chemical space and benchmark metrics.
- **Chemical Space Explorer:** Interactive **UMAP** projection of the dataset.
- **3D Inspector:** Generates and visualizes MMFF94 optimized conformers.

### Run with Docker Compose:
```bash
docker-compose up --build
```
- **API & Premium UI:** `http://localhost:8000`
- **Streamlit Analytical Dashboard:** `http://localhost:8501`

---

## 🤖 Model Zoo

### Baselines (Fingerprint-based)
| Model | Features | Library | Performance |
|-------|----------|---------|-------------|
| **Random Forest** | Morgan FP (2048-bit) | scikit-learn | Strong baseline for small datasets |
| **XGBoost** | Morgan FP (2048-bit) | XGBoost | Best for high-dimensional fingerprints |

### Graph Neural Networks
| Model | Architecture | Library | Purpose |
|-------|-------------|---------|---------|
| **GCN** | Graph Convolutional Network | PyTorch Geometric | Baseline graph connectivity |
| **GAT** | Graph Attention Network | PyTorch Geometric | Attention-based node importance |
| **MPNN** | Message Passing Neural Network | PyTorch Geometric | Edge-conditioned message passing |
| **Multi-Task GNN** | Shared Backbone + Head | PyTorch Geometric | Collaborative learning across tasks |

**Architecture Details:**
- **Backbone:** 3–5 message passing layers (GraphConv/GATConv/MessagePassing).
- **Uncertainty:** MC Dropout active during inference for variance estimation ($ \sigma $).
- **Readout:** Global Mean Pooling → 2-layer MLP Head.

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/jitesh523/Molecular-Property-Prediction-System..git
cd Molecular-Property-Prediction-System.

# Setup with Conda (recommended)
conda env create -f environment.yml
conda activate molprop

# Or via Virtualenv
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Data Pipeline

```bash
# Download and Standardize benchmark sets
python scripts/download_molnet_datasets.py
python -c "from molprop.data.processor import process_all_benchmark_datasets; from pathlib import Path; process_all_benchmark_datasets(Path('data/raw'), Path('data/processed'))"
```

### 3. Training & Evaluation

```bash
# Train GAT on Blood-Brain Barrier dataset
python scripts/train_gnn.py model=gat dataset=bbbp

# Run Ablation Study
python scripts/run_ablation.py --dataset delaney --task regression
```

### 4. Interactive Dashboard

```bash
streamlit run scripts/portfolio_dashboard.py
```

---

## 📊 Ablation Study

Structured comparison of representation modalities (fingerprints vs descriptors vs graphs vs hybrid).
- **Finding:** GNNs often outperform fingerprints on BBBP due to better handling of spatial connectivity, while Random Forest remains competitive on small solubility datasets (Delaney).

---

## 🔬 Interpretability & Diagnostics

- **Global/Local SHAP:** Explainability for fingerprint-based models.
- **GNNExplainer:** Identifying critical subgraphs (atoms/bonds) for graph-based predictions.
- **Ucertainty (MC Dropout):** Identifying out-of-distribution molecules where predictions are less reliable.
- **Chemical Bias Analysis:** Diagnostic scripts in `scripts/analyze_errors.py` to detect "difficult" chemical scaffolds.

---

## 🛠️ MLOps Stack

| Component | Tool | Industrial Purpose |
|-----------|------|--------------------|
| **Tracking** | MLflow | Lineage of hyperparameters, metrics, and models. |
| **Versioning** | DVC | Git-compatible data and artifact versioning. |
| **Configuration**| Hydra | Composable YAML for reproducible experiments. |
| **Deployment** | Docker | Consistent runtime environments for API/Dashboard. |
| **CI/CD** | Actions | Automated linting (Ruff) and unit testing (Pytest). |

---

## 📁 Repository Structure

```
molprop-prediction/
├── .github/workflows/          # CI + Docker build workflows
├── configs/                    # Hydra YAML configs
├── data/                       # DVC-tracked raw/processed datasets
├── notebooks/                  # Educational walkthrough (00-04)
├── scripts/                    # Training, HPO, and dashboard scripts
├── src/molprop/                # Core source code
├── tests/                      # pytest test suite with coverage
├── results/                    # Benchmarks, ablation, explanations
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # Orchestration for API + Dashboard
└── pyproject.toml              # Project metadata & dependencies
```

---

## ✅ Reproducibility Checklist

- [x] **Pinned dependencies** via `pyproject.toml`
- [x] **Deterministic splits** using scaffold-based splitting
- [x] **Canonical standardization** preserving original ↔ standardized SMILES mapping
- [x] **Experiment logging** with MLflow
- [x] **Data versioning** with DVC pipeline definitions
- [x] **Automated CI** with GitHub Actions
- [x] **Containerized inference** via multi-stage Docker builds

---

## 📝 Resume Bullets

> **Built an end-to-end molecular property prediction platform** integrating ChEMBL 36 and PubChem BioAssay (PUG-REST) with benchmark MoleculeNet datasets; implemented deterministic chemical standardization (salt stripping, canonical SMILES, unit normalization) and dataset versioning.

> **Benchmarked RandomForest and XGBoost baselines against GNN architectures** (GCN, GAT, MPNN) in PyTorch Geometric using scaffold-split cross-validation; conducted structured ablation studies (fingerprint vs descriptors vs graph vs hybrid) and reported RMSE/MAE for regression and ROC-AUC/PR-AUC/MCC for imbalanced classification tasks.

> **Implemented multi-task GNN training** with NaN-masked loss functions supporting mixed regression + classification endpoints, enabling simultaneous prediction of physicochemical and ADMET properties from a single shared backbone.

> **Delivered production-ready MLOps**: experiment tracking (MLflow), reproducible environments, CI with coverage reporting, Optuna hyperparameter optimization, and a Dockerized FastAPI inference service with batch prediction, Swagger docs, and explainability artifacts (SHAP + GNNExplainer).

---

