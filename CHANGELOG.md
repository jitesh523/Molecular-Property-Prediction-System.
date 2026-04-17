# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
