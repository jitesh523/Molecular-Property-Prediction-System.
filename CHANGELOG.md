# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-05-02

### Added
- `veber_filter()` in `standardize.py` — evaluates Veber's oral bioavailability rules (RotatableBonds ≤ 10, TPSA ≤ 140 Å²); returns property values and violation list (Veber et al., J. Med. Chem. 2002).
- `ghose_filter()` in `standardize.py` — evaluates Ghose drug-likeness criteria (−0.4 ≤ LogP ≤ 5.6, 160 ≤ MW ≤ 480, 40 ≤ MR ≤ 130, 20 ≤ NumAtoms ≤ 70); Ghose et al., J. Comb. Chem. 1999.
- `GET /druglikeness` REST endpoint — comprehensive drug-likeness panel running Lipinski Ro5 + Veber + Ghose in a single call with `overall_drug_like` summary flag.
- `feature_importances()` method on `BaselineModel` — returns a dict of feature name → importance score (sorted descending), supporting both RandomForest and XGBoost; raises `AttributeError` if model not yet fitted.
- `tests/test_baselines.py` — 20 unit tests for `BaselineModel` covering RF/XGBoost train, predict, evaluate, cross_validate, save/load, feature_importances, and error handling (entirely missing before).
- 10 new Veber/Ghose tests in `test_standardize.py`.
- 4 new `/druglikeness` integration tests in `test_inference.py`.
- API version bumped to `2.1.0` in the FastAPI app metadata.

## [1.4.0] - 2026-05-01

### Added
- `tanimoto_similarity(smiles1, smiles2)` in `fingerprints.py` — computes Morgan fingerprint-based Tanimoto (Jaccard) similarity between two molecules (ECFP4, 2048 bits); returns `None` for invalid SMILES.
- `POST /compare` REST endpoint — side-by-side molecule comparison returning physicochemical descriptors, Lipinski Ro5 results, and Tanimoto similarity for two input SMILES; gracefully handles one-sided invalid inputs.
- `specificity` and `balanced_accuracy` metrics in `compute_metrics()` for classification tasks (derived from confusion matrix via sklearn).
- `pearson_r` metric in `compute_metrics()` for regression tasks (scipy `pearsonr`).
- `tests/test_evaluate.py` — 13 new unit tests covering `compute_metrics()` for both classification (perfect classifier, random baseline, threshold sensitivity, specificity, type checks) and regression (perfect regressor, Pearson R range/anticorrelation, RMSE ≥ MAE, type checks) scenarios plus invalid-task error handling.
- 7 new `TestTanimotoSimilarity` tests in `test_featurizers.py` (identity, range, symmetry, invalid SMILES, dissimilarity).
- 5 new `/compare` integration tests in `test_inference.py` (valid pair, identical molecules, descriptor count, Lipinski presence, one-sided invalid SMILES).

## [1.3.1] - 2026-04-30

### Added
- `configs/model/gin.yaml` — GIN model config for Hydra CLI (`model=gin` was silently broken).
- `configs/dataset/freesolv.yaml`, `lipo.yaml`, `tox21.yaml` — three new dataset configs covering hydration free energy, lipophilicity, and 12-assay toxicology.
- `GET /search` endpoint — standalone GNN embedding-based KNN similarity search in the vector store (returns top-k neighbours with cosine score).
- GIN backbone support in `scripts/train_gnn.py` (the main generic training script was missing GIN).
- GIN backbone comment in `configs/model/multitask.yaml`.
- 10 new API integration tests in `test_inference.py` covering `/descriptors`, `/lipinski`, `/conformer`, `/generate/status`, and `/search`.

### Fixed
- `benchmarks.yml` CI trigger now includes `scripts/train_gin.py` and `configs/**` path globs so benchmark runs fire on config changes.

## [1.3.0] - 2026-04-30

### Added
- `passes_lipinski_ro5()` in `standardize.py` — returns MW, LogP, HBD, HBA and violation list for oral-bioavailability filtering.
- `generate_multiple_conformers()` + `get_conformer_rmsd()` in `conformers.py` — ETKDGv3 ensemble generation with RMSD-based pruning and pairwise RMSD diagnostics.
- `/conformer` REST endpoint — generates and returns an MMFF94-optimized 3D PDB block for any SMILES, ready for py3Dmol visualisation.
- `/lipinski` REST endpoint — evaluates Lipinski Ro5 for any SMILES via a simple GET request.
- `save()` / `load()` persistence methods on `BaselineModel` using joblib.
- `cross_validate()` method on `BaselineModel` — stratified k-fold (classification) or k-fold (regression) CV with per-fold metric logging.
- GIN backbone option in `MultiTaskGNN` (was previously missing).
- `scripts/train_gin.py` — dedicated GIN training script with AdamW + cosine-annealing LR, gradient clipping, and full `compute_metrics()` test reporting.
- Tests for `stratified_split` (coverage, class balance, determinism) and `scaffold_kfold` (fold count, coverage, no-overlap) in `test_splits.py`.
- Lipinski Ro5 tests in `test_standardize.py`.

### Fixed
- **Bug**: `recreate_collection()` deprecated in qdrant-client ≥ 1.9 — replaced with idempotent `create_collection()` + `collection_exists()` guard.
- Added `count()` and `delete_collection()` utility methods to `MolecularVectorStore`.

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
