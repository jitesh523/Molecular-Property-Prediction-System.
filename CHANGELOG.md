# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.42.0] - 2026-05-22

### Added — Bemis–Murcko scaffold clustering

- **🌳 `POST /scaffold/cluster`** backed by `src/molprop/features/scaffold_cluster.py`:
  - Groups a list of SMILES by their Bemis–Murcko scaffold.
  - `generic=true` collapses atoms→C and bonds→single (generic Murcko framework), useful for grouping isosteres / bioisosteres (e.g. pyridine ≡ benzene).
  - `min_cluster_size` filter for dropping singleton scaffolds.
  - Clusters are deterministically sorted by descending size.
- **🧪 8 new tests** covering empty input, benzene-analogue collapse, invalid-SMILES handling, generic-framework isostere merging, size-filter, deterministic ordering, and the endpoint round-trip.
- SDK helper `client.scaffold_cluster(smiles_list, generic=…, min_cluster_size=…)`.

## [2.41.0] - 2026-05-22

### Added — Free-Wilson SAR UI tab

- **📐 Free-Wilson tab** wired to `POST /freewilson`:
  - Core SMARTS/SMILES input + multi-line `SMILES, activity` textarea (TAB or comma).
  - Configurable `min_occurrences` filter.
  - KPI cards: used rows / intercept / R² / RMSE.
  - **R-group contributions table** sorted by |Δ| with green ▲ positive / red ▼ negative arrows and reference-occupant marker.
  - **Per-molecule predictions table** showing observed vs. predicted with colour-coded residuals.

## [2.40.0] - 2026-05-21

### Added — Depict tab, Prometheus metrics, Free-Wilson SAR

- **🎨 v2.37.0 — Standalone Depict UI tab** — fully-featured 2D rendering tab:
  - SMILES + optional highlight-SMARTS inputs, configurable width/height.
  - Live KPI showing the canonical SMILES the server rendered + count of highlighted atoms.
  - "💾 Download SVG" button saves a publication-ready vector file.
- **📊 v2.38.0 — Prometheus metrics endpoint** — `GET /metrics/prometheus`:
  - Standard text exposition format (`version=0.0.4`) consumable by Prometheus / Grafana Agent / VictoriaMetrics.
  - Emits `molprop_uptime_seconds`, `molprop_requests_total{route=…}`, `molprop_request_errors_total{route=…}`, `molprop_request_latency_seconds_sum{route=…}`, plus `molprop_cache_{hits,misses,size}` from the in-process endpoint cache.
  - Path is exempt from the rate limiter so scrapers never trip a 429.
- **📐 v2.39.0 — Free-Wilson SAR analysis** — new `POST /freewilson` backed by `src/molprop/features/freewilson.py`:
  - Decomposes a series of analogues around a common core, fits a least-squares additive model where each R-group occupant becomes a one-hot feature.
  - Returns intercept, R², RMSE, per-occupant additive contributions, and per-molecule observed/predicted/residual values.
  - Uses the Moore–Penrose pseudo-inverse so it handles rank-deficient designs (small or sparse series).
  - Numpy-only — zero new dependencies.
- **🧪 v2.40.0 — Free-Wilson SDK + CLI + tests** —
  - `client.free_wilson(core, smiles_list, activities, min_occurrences=…)` SDK helper.
  - `molprop freewilson --core <smarts> data.tsv` CLI command (TAB or comma-separated SMILES + activity).
  - 6 new tests covering invalid-core handling, length validation, perfect additive recovery (R²≈1, RMSE≈0), `min_occurrences` filtering, and the Prometheus exposition format. **30 cheminformatics tests passing in 7.9s.**

## [2.36.0] - 2026-05-21

### Added — Live 2D depiction in Predict tab, rate limiting, browser SDK

- **🎨 v2.33.0 — Live 2D depiction in Predict tab** — every successful prediction now auto-renders a 2D SVG of the standardized molecule beside the Inference Results panel:
  - Server-cached via `@cached_json` (instant on second view).
  - "💾 Download SVG" button saves a publication-ready SVG file named after the canonical SMILES.
  - Reusable `renderDepiction(smiles, highlightSmarts?)` helper exposed for other tabs to call.
- **🛡️ v2.34.0 — In-process token-bucket rate limiter** (`src/molprop/serving/rate_limit.py`)
  - Per-IP `TokenBucket` with smooth refill — no extra dependencies.
  - Configurable via env: `MOLPROP_RATE_LIMIT` (default 120), `MOLPROP_RATE_WINDOW` (default 60s), `MOLPROP_RATE_DISABLE`.
  - Honours `X-Forwarded-For` for first-hop IP behind a reverse proxy.
  - Exempts health/version/cache/openapi/docs paths so monitoring never trips the limit.
  - Adds `X-RateLimit-Limit` / `X-RateLimit-Remaining` headers to every response, returns `429` + `Retry-After` on overflow.
- **📘 v2.35.0 — Browser / TypeScript SDK** (`docs/BROWSER_SDK.md`)
  - Single-file zero-dependency client (TS and JS variants, ≈80 lines each).
  - Examples: drop-in `<script>` for plain HTML, Node 18+ / Deno / Bun ESM import, and a React `<MoleculePreview>` component that calls `/depict` as the user types.
  - Documents `MolpropAPIError`, `429` rate-limit handling, and `X-RateLimit-*` headers.
- **🧪 v2.36.0 — Rate-limit tests** (`tests/test_rate_limit.py`)
  - 7 new tests: token-bucket capacity & refill semantics, `429` response shape, header propagation, exempt paths, and per-IP bucket segregation via `X-Forwarded-For`. **Total: 24 cheminformatics tests, all passing in 2.27s.**

## [2.32.0] - 2026-05-20

### Added — Reaction & MMP UI tabs, endpoint caching, more tests, 2D depiction

- **🧪 v2.29.0 — Reactions & MMP UI tabs** — two new standalone tabs:
  - **Reactions tab** wired to `POST /react`: dropdown auto-populated from `GET /react/named`, custom SMARTS box, multi-line substrate-tuples textarea, KPI cards (input sets / produced / unique), product chip cloud.
  - **MMP tab** wired to `POST /mmp`: multi-line analogue input, configurable max-substituent-atoms and max-pairs caps, summary KPI cards (valid molecules / shared contexts / pairs), and a full pair table with R<sub>A</sub>, R<sub>B</sub>, and colour-coded Δheavy.
- **⚡ v2.30.0 — In-process TTL+LRU cache** (`src/molprop/serving/cache.py`)
  - Thread-safe LRU+TTL cache with hit/miss accounting.
  - `@cached_json(ttl_seconds=600)` decorator inspects Pydantic request models and uses a `sha256` JSON hash as the cache key.
  - Applied to 8 stateless cheminformatics endpoints: `/scaffold`, `/isomers`, `/standardize`, `/functional_groups`, `/mcs`, `/rgroups`, `/mmp`, `/alerts`.
  - New admin endpoints `GET /cache/stats` and `POST /cache/clear`.
- **🧪 v2.31.0 — Tests for reactions, MMP, and cache** (`tests/test_reactions_mmp_cache.py`)
  - 17 tests covering: amide coupling product correctness, N-methylation single-reactant case, named-catalog integrity, MMP pair finding on a phenol series, name-validation guards, LRU eviction, TTL expiry, cache-stats accounting, and bypass for non-serialisable arguments.
- **🎨 v2.32.0 — 2D SVG depiction** — `POST /depict`:
  - Renders a 2D SVG of any SMILES via `rdMolDraw2D.MolDraw2DSVG`.
  - Optional atom highlighting either by explicit indices or by SMARTS match.
  - Configurable width/height; result is cached via `@cached_json`.
  - Exposed in the SDK (`client.depict`) and CLI (`molprop depict <smiles> -o file.svg`).

## [2.28.0] - 2026-05-20

### Added — R-group UI, reactions, examples, Matched Molecular Pairs

- **🧩 v2.25.0 — R-group decomposition UI tab** — new standalone tab wired to `POST /rgroups`:
  - Core SMARTS/SMILES input + multi-line analogue textarea.
  - KPI cards (matched / unmatched / R-positions) + unique-R-groups-per-position panel + per-molecule assignment table.

- **⚗️ v2.26.0 — Reaction SMARTS application** — new `POST /react` and `GET /react/named` backed by `src/molprop/features/reactions.py`:
  - Accepts either a raw reaction SMARTS or a `named:` key from a built-in catalog of common reactions: `amide_coupling`, `ester_hydrolysis`, `n_methylation`, `alcohol_to_aldehyde`, `suzuki_coupling`, `nitro_reduction`.
  - Substrates are passed as reactant-tuples to support multi-reactant transformations.
  - Returns per-input product sets + a flat list of unique canonical product SMILES.
  - Exposed in the SDK (`client.react`, `client.react_named`) and CLI (`molprop react`, `molprop react-list`).

- **📚 v2.27.0 — Worked-example walkthrough** — new `examples/` directory:
  - `examples/01_sdk_walkthrough.py` — end-to-end Python tour exercising health, scaffold + SA, functional groups, compare + MCS, R-groups, reaction SMARTS, structural alerts, library CRUD, and the aggregated Markdown report against a panel of common drugs (aspirin, acetaminophen, ibuprofen, caffeine, ethanol).
  - `examples/README.md` — accompanying recipe book with CLI mini-recipes (PAINS triage, substructure scan, batch reporting, reaction batch, named-reaction catalog).

- **🔗 v2.28.0 — Matched Molecular Pairs (MMP)** — new module `src/molprop/features/mmp.py` implementing single-cut MMP analysis (Hussain & Rea 2010, JCIM 50, 339):
  - Fragments each molecule along every non-ring acyclic single bond.
  - Groups molecules by shared "context" fragment.
  - Emits all unordered pairs with different substituents, including R_A, R_B, and Δheavy_atoms per pair.
  - `POST /mmp` endpoint with configurable `max_substituent_atoms` and `max_pairs` caps.
  - Exposed in the SDK (`client.mmp`) and CLI (`molprop mmp <smiles_file>`).

## [2.24.0] - 2026-05-19

### Added

- **🛡️ v2.21.0 — Verified CI / pre-commit** — confirmed existing `.github/workflows/ci.yml` (matrix Python 3.11/3.12 + ruff + bandit + pytest-cov) and `.pre-commit-config.yaml` (ruff + ruff-format + standard hygiene hooks). No code changes — documenting that they remain green with all new modules.
- **📝 v2.22.0 — README modernisation** — new top-of-README "What's New in v2.x" section: tabular endpoint index (30+ endpoints), 14 UI tabs listed with highlights, copy-pasteable SDK + CLI snippets, test-suite invocation. Makes the cheminformatics suite discoverable for portfolio reviewers.
- **🔎 v2.23.0 — Substructure tab** — new standalone UI tab for `POST /substructure`:
  - SMARTS / SMILES query box, optional project scope, configurable limit.
  - Collapsible textarea to paste a custom candidate list instead of searching the saved library.
  - Sortable results table with rank, SMILES, name, and hit-count badge.
- **🧬 v2.24.0 — R-group decomposition** — new module `src/molprop/features/rgroups.py` wrapping RDKit's `rdRGroupDecomposition`:
  - `POST /rgroups` accepts a core (SMARTS / SMILES) plus an analogue list, returns per-molecule R-group assignments, unique R-groups per position, and matched / unmatched counts.
  - Exposed in the SDK (`client.rgroups`) and the CLI (`molprop rgroups <core> <smiles_file>`).

## [2.20.0] - 2026-05-19

### Added — CLI, MCS, structural alerts, library CSV I/O, API integration tests

- **🖥️ v2.17.0 — Command-line interface** (`src/molprop/cli.py`)
  - `molprop` entry-point registered in `pyproject.toml`.
  - Click-based CLI wrapping every SDK method with typed flags and a `--pretty/--compact` JSON output toggle.
  - Subcommands: `health`, `version`, `predict`, `scaffold`, `fg`, `isomers`, `compare`, `mcs`, `alerts`, `standardize`, `admet`, `substructure`, `report` (with `-o file.md`), and a `library` subgroup (`save`, `list`, `get`, `delete`, `projects`).
  - Uniform `MolpropAPIError` → coloured stderr + non-zero exit code.

- **🔗 v2.18.0 — Maximum Common Substructure + Library CSV I/O**
  - New module `src/molprop/features/mcs.py` wrapping `rdFMCS.FindMCS` with ring-aware defaults.
  - New `POST /mcs` returns SMARTS, atom/bond counts, fractional coverage of each molecule, and matching atom indices.
  - New `GET /library/export/csv` streams the (optionally project-filtered) library as a CSV download.
  - New `POST /library/import` bulk-upserts up to 10,000 rows with per-row error reporting.

- **🚨 v2.19.0 — Structural Alerts** (PAINS, Brenk, NIH, ZINC)
  - New `POST /alerts` endpoint backed by RDKit's `FilterCatalog`.
  - Configurable catalogs: `PAINS`, `PAINS_A`, `PAINS_B`, `PAINS_C`, `BRENK`, `NIH`, `ZINC`.
  - Reports each flagged substructure with its catalog name, description, and atom-pair indices for highlighting; surfaces an `is_clean` boolean for quick filtering.

- **🧪 v2.20.0 — API integration tests** (`tests/test_api_integration.py`)
  - End-to-end tests via FastAPI's `TestClient` covering scaffold, functional groups, isomers, substructure, compare, standardize, MCS, alerts, report, and a full Library CRUD round-trip (incl. CSV import/export).
  - 14 tests; ML-heavy endpoints are deliberately omitted to keep the suite CI-friendly.

### Polish
- **UI wiring** — Compare tab now also shows the **🔗 Maximum Common Substructure** panel (SMARTS, atom/bond counts, fractional coverage of each molecule). Scaffold tab now also shows the **🚨 Structural Alerts** panel (PAINS / Brenk / NIH) with green "clean" state and red/orange/yellow severity-coloured hit cards.
- **Documentation** — new `docs/CLI_AND_SDK.md` with full SDK + CLI reference, end-to-end example pipeline, and a method index.

## [2.16.0] - 2026-05-18

### Added — UI completion, test coverage, and a Python SDK

- **🔬 v2.13.0 — Compare tab UI** — wires the `/compare` endpoint to a fully interactive panel: large MACCS Tanimoto headline with color coding (green ≥ 0.7 / yellow ≥ 0.4 / red < 0.4), scaffold-equality banner, side-by-side molecule cards (canonical SMILES, Lipinski / Veber / Ghose tick marks, SAScore, Murcko), and a full descriptor-delta table with ▲/▼ arrows.
- **🧬 v2.14.0 — Isomers tab UI** — wires the `/isomers` endpoint: configurable max-tautomer / max-stereoisomer caps, canonical-tautomer hero panel, separate accent-colored card grids for tautomers (purple) and stereoisomers (cyan) with "CANONICAL" badges and copy-to-clipboard.
- **🧪 v2.15.0 — Test suite** — two new pytest modules:
  - `tests/test_storage.py` — 9 tests covering `CompoundLibrary` CRUD, upsert idempotency, project/tag filters, persistence across reopens.
  - `tests/test_cheminformatics_features.py` — 18 tests covering scaffolds (aspirin/quinine/macrocycle), functional groups (aspirin / acetaminophen / atom-index validity), isomers (canonical-flag uniqueness, limit enforcement), and substructure search (SMARTS + SMILES queries, invalid-input handling, limit respect).
- **🐍 v2.16.0 — Python SDK** (`src/molprop/client.py`) — `MolpropClient` with one method per endpoint:
  - `health`, `version`, `predict`, `predict_batch`, `generate`, `generate_smart`, `admet`, `admet_batch`, `scaffold`, `scaffold_batch`, `functional_groups`, `isomers`, `substructure`, `compare`, `standardize`, `conformer`, `search_similar`, `report`, `library_save`, `library_list`, `library_get`, `library_update`, `library_delete`, `library_projects`.
  - Uniform error path via `MolpropAPIError` (carries status code + detail + URL).
  - Reusable `requests.Session`, configurable base URL & timeout.

## [2.12.0] - 2026-05-17

### Added — 4 new cheminformatics endpoints (one batch release)

- **🧬 v2.9.0 — Isomer Enumeration** (`POST /isomers`)
  - New module `src/molprop/features/isomers.py`.
  - Enumerates **tautomers** (RDKit `TautomerEnumerator`) and **stereoisomers** of unassigned centres (RDKit `EnumerateStereoisomers`).
  - Returns the canonical tautomer + every enumerated isomer, capped configurably (defaults: 25 tautomers, 16 stereoisomers).

- **🔎 v2.10.0 — Substructure SMARTS Search** (`POST /substructure`)
  - New module `src/molprop/features/substructure.py`.
  - Accepts SMARTS or SMILES query patterns; falls back gracefully between the two.
  - Searches an explicit candidate list **or the persistent compound library** (optionally scoped to a project) — bridges Library + cheminformatics.
  - Returns matching SMILES with hit counts and atom-index lists for substructure highlighting.

- **🔬 v2.11.0 — Compound Comparison** (`POST /compare`)
  - Side-by-side analysis of two molecules: canonical SMILES, MACCS Tanimoto similarity, descriptor diffs (per-descriptor delta), Lipinski / Veber / Ghose flags, SAScore, Bemis–Murcko scaffold equality.

- **🧹 v2.12.0 — Standardization Report** (`POST /standardize`)
  - Detailed standardisation breakdown: salt stripping, neutralisation (Uncharger), largest-fragment selection, and the final canonical form.
  - Reports whether each step changed the input — useful for QA on user-submitted compounds before storing them in the library.

## [2.8.0] - 2026-05-16

### Added — 📄 Aggregated Markdown Report
- New `POST /report` endpoint that bundles standardisation, drug-likeness filters (Lipinski / Veber / Ghose), descriptors, ADMET, scaffold + SAScore, and functional groups into a single Markdown document.
- "📄 Download Report" button in the Scaffold tab — produces `report_<smiles>_<date>.md`.
- Configurable sections via request body flags (`include_admet`, `include_scaffold`, `include_functional_groups`, `include_descriptors`).

## [2.7.0] - 2026-05-16

### Added — ⚗️ Functional Group Annotator
- New module `src/molprop/features/functional_groups.py` with `detect_functional_groups()` and a curated catalog of **~40 SMARTS patterns** spanning carbonyl, amine, nitrogen, oxygen, sulfur, phosphorus, halogen, aromatic, heteroaromatic, cyclic, and aliphatic categories.
- Returns per-group hit count + matching atom indices (suitable for substructure highlighting in 2D depictions).
- New endpoint `POST /functional_groups`.
- Integrated into the **Scaffold tab**: category chips with color coding, sortable per-group cards showing name, count, SMARTS pattern, and category-tinted accent.

## [2.6.0] - 2026-05-16

### Added — 🦴 Scaffold Analysis & Synthetic Accessibility
- New module `src/molprop/features/scaffolds.py` with `analyze_scaffold()` returning a `ScaffoldAnalysis` dataclass.
- **Bemis–Murcko scaffold** extraction (rings + linkers) and **Generic Murcko framework** (atoms reduced to C, bonds to single).
- Ring topology metrics: total rings, aromatic / aliphatic split, largest ring size, spiro atoms, bridgehead atoms, macrocycle detection.
- **Synthetic Accessibility Score** (1.0 easy → 10.0 hard): combines ECFP4 fragment diversity, heavy-atom log-size penalty, stereocenter / spiro / bridgehead / macrocycle penalties. Lightweight reimplementation of the Ertl–Schuffenhauer heuristic — no contrib data file needed.
- New endpoints: `POST /scaffold` and `POST /scaffold/batch`.
- New **🦴 Scaffold tab** in UI: gradient SAScore meter (green→yellow→red) with class badge, side-by-side scaffold display, 7-card ring topology grid.

## [2.5.0] - 2026-05-16

### Added — 📚 Compound Library (persistent storage)
- New module `src/molprop/storage/library.py` — thread-safe SQLite-backed CRUD using stdlib `sqlite3` only (no new dependencies).
- Schema: `compounds(id, smiles, name, project, tags, properties_json, notes, created_at, updated_at)`. Idempotent upsert on `(smiles, project)`.
- 6 new REST endpoints under `/library`:
  - `POST /library` — save / upsert compound
  - `GET /library` — list with filters (project, tag, search, limit, offset)
  - `GET /library/projects` — distinct projects + tags
  - `GET /library/{id}` — fetch one
  - `PATCH /library/{id}` — partial update
  - `DELETE /library/{id}` — remove
- New **📚 Library tab** in UI: save form (SMILES, name, project, tags, notes), live filter bar (search debounce 300ms, project & tag dropdowns), sortable table with per-row Predict / Delete actions.
- "Save to Library" button auto-injected into the Predict result panel.
- DB lives at `data/library.db` (gitignored).

## [2.4.0] - 2026-05-15

### Added — 5 major UI/API upgrades
- **📁 Batch Predictions tab** — Paste multiple SMILES (one per line) or upload a CSV file (auto-detects SMILES column). Returns sortable table with per-row "Use" action and CSV export.
- **🔍 Fingerprint Similarity Search** — `POST /search/similar` endpoint using Morgan (ECFP, 2048-bit) or MACCS fingerprints with Tanimoto coefficient. Configurable top-K and threshold. Color-coded similarity badges in results grid.
- **🧬 3D Structure Viewer** — Interactive 3Dmol.js viewer with conformer generation (RDKit ETKDGv3 + MMFF94). Stick / Ball-and-Stick / Space-Fill / Line render styles. Spin animation, reset view, and PDB download.
- **📊 Analytics Dashboard** — KPI cards (total predictions, mean, min/max range, avg uncertainty) and vanilla canvas charts: prediction distribution histogram (12 bins, gradient-filled) and timeline trend chart. Auto-refreshes from `localStorage` history.
- **🎯 Smart Generation** — `POST /generate/smart` endpoint samples molecules and filters by property ranges (MW, LogP, QED, TPSA). Reports acceptance rate and supports up to 2000 attempts per request.

### Other
- Custom events `model-ready` and `vae-ready` to unify button activation.
- Auto-prefill of SMILES inputs across tabs from last predicted molecule.

## [2.3.0] - 2026-05-14

### Added
- **🧪 ADMET Property Prediction** — `POST /admet` and `POST /admet/batch` endpoints.
  - **Absorption**: Lipinski Ro5, Veber rules, TPSA, oral bioavailability classification
  - **Distribution**: BBB permeability (Clark model), fraction Csp3
  - **Metabolism**: CYP3A4 / CYP1A2 / CYP2D6 inhibition risk flags
  - **Excretion**: Renal clearance estimate from LogP
  - **Toxicity**: hERG cardiotoxicity risk, Ames mutagenicity, PAINS alerts, Brenk structural alerts
  - Composite 0-100 overall score (≥60 = pass)
- **🧪 ADMET tab** in the web UI with 5 property cards (color-coded pass/warn/fail), alert list, overall badge.
- Auto-prefill ADMET SMILES input from the last predicted molecule.
- `src/molprop/features/admet.py` — `compute_admet()` function, `ADMETResult` dataclass.

## [2.2.0] - 2026-05-14

### Added
- **🌐 Latent Space Explorer tab** — interactive PCA 2D projection of the VAE latent space.
  - `POST /latent_map` endpoint: samples molecules, encodes to latent space, projects to 2D via PCA (no extra dependencies).
  - Canvas scatter plot colored by QED — purple (low) → green (high).
  - Hover tooltip shows QED, MW, and SMILES.
  - Click any dot to select and decode the molecule.
  - Selected molecule can be sent directly to Predict or Optimize tabs.
  - Optional seed SMILES highlighted with ⭐ on the map.
- `LatentMapRequest`, `LatentMapPoint` models in API.

## [2.1.0] - 2026-05-14

### Added
- **📋 Prediction History tab** — session log of all predicted molecules backed by `localStorage` (up to 50 entries).
  - Each entry shows SMILES, prediction value, uncertainty, timestamp, and property chips.
  - Per-row "Predict" and "Optimize" shortcut buttons.
- **⚖️ Side-by-side comparison** — select 2+ molecules with checkboxes and compare all properties in a table.
- **CSV export** for entire history log.
- **🗑 Clear history** button.
- History auto-renders when switching to History tab.

## [2.0.0] - 2026-05-14

### Added
- **Multi-Objective Pareto Optimization** — `POST /optimize/pareto` endpoint finds molecules simultaneously optimal across all selected objectives (NSGA-II inspired).
  - Supported objectives: `qed`, `neg_sas`, `logp_norm`, `mw_norm`, `tpsa_norm`, `hbd_norm`, `hba_norm`
  - Implements Pareto dominance and crowding distance for diversity preservation
  - Supports seed molecule for neighborhood Pareto exploration
- **`ParetoOptimizer` class** in `models/pareto.py` with `dominates()`, `compute_pareto_front()`, `compute_crowding_distance()` helpers
- **🏆 Pareto Mode** section in the Optimize UI with objective checkboxes and dedicated results panel
- **🔬 Find Analogs** button in Predict tab — one-click bridge that pre-fills the Optimize seed field with the currently predicted molecule
- CSV export for Pareto front results
- Checkbox-label styling for objective selectors

## [1.9.0] - 2026-05-13

### Added
- **QED (Drug-likeness) and SAS (Synthetic Accessibility)** property targets for optimization.
  - QED: 0.0-1.0 scale, higher is more drug-like (RDKit-based).
  - SAS: 1.0-10.0 scale, lower is easier to synthesize.
- **Seed molecule support** — start optimization from a known molecule using its SMILES string.
  - `seed_smiles` parameter in `POST /optimize` request.
  - VAE encodes the seed to latent space and explores nearby regions.
- **CSV export** in the UI — download optimized molecules as CSV with all computed properties.
- UI enhancements:
  - Property range inputs for QED and SAS with helpful hints.
  - Seed molecule input field with placeholder example.
  - Export button in optimization results panel.

### Changed
- `LatentOptimizer.optimize()` now accepts optional `seed_smiles` parameter.
- Both `gradient_ascent` and `random_walk` methods support seed-based optimization.

## [1.8.0] - 2026-05-12

### Added
- **Guided Molecular Optimization** — new `POST /optimize` endpoint that navigates the VAE latent space to discover molecules matching target property constraints (MW, LogP, TPSA, HBD, HBA).
- Two optimization methods:
  - `gradient_ascent`: Uses finite-difference gradients to optimize latent vectors toward target properties.
  - `random_walk`: Baseline Monte Carlo sampling with selection pressure.
- `LatentOptimizer` class in `models/optimization.py` — handles latent space traversal, property scoring, and SMILES decoding.
- New "🎯 Optimize" tab in the web UI with property range inputs, method selection, and candidate display.
- CSS styling for range controls and select dropdowns.
- Tests for `/optimize` endpoint covering VAE availability, invalid methods, and valid request handling.

### Fixed
- Added standard `background-clip` property alongside `-webkit-background-clip` for CSS compatibility.

## [1.7.0] - 2026-05-11

### Added
- `GET /version` — returns package version, API version, and Torch runtime info.
- `GET /metrics` — lightweight in-process request counters, error counts, average latency per route, and uptime.
- Example-molecule dropdown, copy-to-clipboard button for standardized SMILES, and server-latency display in the web UI.
- Developer `Makefile` with `install`, `dev`, `lint`, `format`, `test`, `cov`, `api`, and Docker targets.
- Tests for `/version`, `/metrics`, and empty-batch validation on `/predict/batch`.

### Changed
- CI now runs a Python `3.11`/`3.12` matrix with workflow concurrency (`cancel-in-progress`) and uploads a coverage artifact on 3.11.
- `POST /predict/batch` now returns `400` on empty `smiles_list` before reaching the model.
- Request-timing middleware also records in-process counters consumed by `/metrics`.

## [1.6.0] - 2026-05-03

### Added
- `dice_similarity()` in `fingerprints.py` — Dice (Sørensen–Dice) molecular similarity via Morgan fingerprints (ECFP4, 2048-bit); complements `tanimoto_similarity()` and is preferred for molecules of unequal size (Dice ≥ Tanimoto, provably).
- `batch_smiles_to_graphs()` in `graphs.py` — missing batch wrapper over `smiles_to_graph()`, consistent with batch functions in `fingerprints.py` and `descriptors.py`; supports optional per-molecule label attachment.
- `GET /scaffold` REST endpoint — returns the Bemis–Murcko scaffold SMILES for a molecule; returns `null` for acyclic compounds; 400 on invalid input.
- `POST /batch/descriptors` REST endpoint — computes 18 ADMET/Lipinski descriptors for up to 100 SMILES in one call; returns per-molecule results with `error` fields for invalid entries and top-level `total`/`failed` counts.
- `TestTemporalSplit` (6 tests) added to `test_splits.py` — `temporal_split` had zero test coverage; tests cover overlap, full coverage, chronological ordering, ratio accuracy, reverse-input, and custom fractions.
- `TestDiceSimilarity` (6 tests) and `TestBatchSmilesToGraphs` (6 tests) added to `test_featurizers.py`.
- 7 new integration tests in `test_inference.py` covering `/scaffold` (4 tests) and `/batch/descriptors` (3 tests).

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
