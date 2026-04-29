# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-04-29

### Added
- `GINModel` (Graph Isomorphism Network) — provably most expressive GNN architecture (Xu et al., ICLR 2019) with JK-sum multi-scale readout and BatchNorm layers.
- `encode()` method and `mc_dropout` support added to `MPNNModel` (previously missing).
- `compute_metrics()` function in `evaluate.py` returning a unified dict of ROC-AUC, AP, Accuracy, F1, MCC (classification) or RMSE, MAE, R², mean/std error (regression).
- `stratified_split()` in `splits.py` for class-balanced train/val/test splits on imbalanced datasets.
- MACCS keys fingerprint (`smiles_to_maccs`, `batch_smiles_to_maccs`) added to `fingerprints.py`.
- 9 additional physicochemical descriptors in `descriptors.py`: NumRings, NumAromaticRings, FractionCSP3, BertzCT, MolMR, NumStereocenters, NHOHCount, NOCount, NumValenceElectrons.
- `/descriptors` REST endpoint in the API — returns 18 ADMET/Lipinski descriptors and optional MACCS fingerprint for any SMILES.
- `GINModel` registered in `load_gnn_model` and `DEFAULT_GNN_CONFIGS`.
- `tests/test_models.py` covering forward pass, `encode()`, `mc_dropout`, batch inference, and determinism for all four GNN architectures.

### Fixed
- **Bug**: `ROOT` path variable was referenced inside `lifespan()` before it was defined (used at module level now).

## [1.1.0] - 2026-04-18

### Added
- `SECURITY.md` to define vulnerability reporting processes.
- GitHub Issue Templates for bug reports and feature requests.
- Technical Whitepaper generation script and baseline results.
- Support for additional MoleculeNet datasets (FreeSolv, Lipophilicity).
- Automated security scanning (Bandit) in CI/CD pipeline.

### Changed
- Refactored `pyproject.toml` with professional metadata and entry points.
- Hardened `ci.yml` with linting and security jobs.

## [1.0.0] - 2026-04-10
- Initial release of the Molecular Property Prediction System.
- Support for GNNs (GCN, GAT, MPNN) and Fingerprint baselines.
- Streamlit dashboard and FastAPI deployment.
