# Technical Portfolio: Molecular Property Prediction System

## Executive Summary
This project implements a professional-grade machine learning platform for predicting biochemical and ADMET properties of small molecules. It was designed to demonstrate proficiency in **ML Engineering**, **Chemical Informatics**, and **MLOps** for a pharmaceutical context (specifically aligned with the Eli Lilly Small Molecule AI Engineer Intern expectations).

## 🚀 Key Features
- **Multi-Source Data Ingestion:** Automated pipelines for MoleculeNet benchmarks, ChEMBL potency data, and PubChem assays.
- **Robust Chemical Informatics:** Deterministic standardization (salt stripping, canonicalization) and multi-modal featurization (Fingerprints, RDKit descriptors, and Molecular Graphs).
- **Advanced Modeling:** Implementation of Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Message Passing Neural Networks (MPNN) in PyTorch Geometric.
- **Industrial Evaluation:** Scaffold-based internal validation to simulate real-world prospective prediction scenarios.
- **Production MLOps:** Containerized deployment (Docker/FASTAPI), experiment tracking (MLflow), and interactive diagnostics (Streamlit).

## 📊 Methodology

### 1. Data Processing & Standardization
Molecules are processed through a strict pipeline to ensure model robustness:
- **Sanitization:** Detecting and logging valid chemical valences.
- **Fragment Management:** Retaining the largest organic fragment to handle mixtures/solvates.
- **Canonicalization:** Generating unique SMILES strings to prevent data leakage during splitting.

### 2. The Model Zoo
- **Baselines:** Random Forest and XGBoost trained on Morgan Fingerprints (2048-bit) provide a solid performance ceiling.
- **GNN Backbone:** A shared message-passing architecture that transforms graph topology into latent embeddings.
- **Multi-Task Heads:** Task-specific MLP heads allow the model to learn shared chemical representations across multiple endpoints while handling missing labels via NaN-masking.

### 3. Uncertainty Quantification
To prevent over-confident predictions on "out-of-distribution" molecules, the system implements **Monte Carlo Dropout**. By keeping dropout active during inference, the model generates a distribution of predictions, allowing the calculation of a confidence interval (standard deviation).

### 4. Applicability Domain
Using **UMAP (Uniform Manifold Approximation and Projection)**, the dashboard allows users to visualize where a new molecule sits in relation to the training set, identifying potential "activity cliffs" or data gaps.

## 🛠️ Tech Stack
- **Core:** Python 3.11+, RDKit, PyTorch, PyTorch Geometric
- **Analytics:** Pandas, NumPy, Scikit-learn, UMAP-learn, Plotly
- **MLOps:** MLflow, DVC, Docker, FastAPI
- **Frontend:** Streamlit

---
**Author:** [Your Name/GitHub]
**Context:** Eli Lilly AI Engineer Internship Portfolio
