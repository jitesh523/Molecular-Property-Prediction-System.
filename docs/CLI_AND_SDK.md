# molprop CLI & Python SDK

This guide covers the two programmatic interfaces shipped with molprop:

1. **`MolpropClient`** — a typed Python SDK (`src/molprop/client.py`)
2. **`molprop` CLI** — a Click-based command-line wrapper (`src/molprop/cli.py`)

Both talk to a running molprop FastAPI server (default: `http://localhost:8000`).

---

## 1. Installation

After `pip install -e .` the following entry points are registered:

| Command            | Purpose                                       |
| ------------------ | --------------------------------------------- |
| `molprop`          | CLI                                           |
| `molprop-train`    | Train a GNN model (existing)                  |
| `molprop-dashboard`| Portfolio dashboard (existing)                |

The Python SDK is importable as `from molprop.client import MolpropClient`.

---

## 2. Python SDK

```python
from molprop.client import MolpropClient, MolpropAPIError

client = MolpropClient("http://localhost:8000", timeout=60)

# Health check
print(client.health())

# Property prediction
result = client.predict("CC(=O)Oc1ccccc1C(=O)O")
print(result["prediction"])

# Cheminformatics
scaf  = client.scaffold("CC(=O)NC1=CC=C(O)C=C1")
fgs   = client.functional_groups("CC(=O)O")
iso   = client.isomers("CC(=O)NC1=CC=C(O)C=C1", max_tautomers=10)
cmp   = client.compare("CCO", "CCN")
mcs   = client.mcs("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1")
flags = client.alerts("Cc1ccc(cc1)/N=N/c2ccc(cc2)N(C)C")  # azo PAINS

# Compound library
saved   = client.library_save("CCO", name="Ethanol", project="solvents",
                              tags=["alcohol", "polar"])
listed  = client.library_list(project="solvents")
hits    = client.substructure("c1ccccc1", project="drug-x")

# One-click Markdown report
md = client.report("CC(=O)NC1=CC=C(O)C=C1")["markdown"]

# Error handling
try:
    client.predict("not a smiles")
except MolpropAPIError as e:
    print(e.status_code, e.detail)
```

### Available methods (24 total)

**Meta:** `health`, `version`
**Prediction:** `predict`, `predict_batch`
**Generation:** `generate`, `generate_smart`
**ADMET:** `admet`, `admet_batch`
**Cheminformatics:** `scaffold`, `scaffold_batch`, `functional_groups`, `isomers`,
`substructure`, `compare`, `mcs`, `alerts`, `standardize`, `conformer`
**Similarity:** `search_similar`
**Reports:** `report`
**Library:** `library_save`, `library_list`, `library_get`, `library_update`,
`library_delete`, `library_projects`

All methods return Python dicts (or `bytes` for binary endpoints). Non-2xx
responses raise `MolpropAPIError(status_code, detail, url)`.

---

## 3. Command-line interface

```bash
# Globals
molprop --url http://staging.example.com --timeout 30 health
molprop --compact predict "CCO"           # JSON one-liner

# Property prediction
molprop predict "CC(=O)Oc1ccccc1C(=O)O"

# Cheminformatics
molprop scaffold "CC(=O)NC1=CC=C(O)C=C1"
molprop fg "CC(=O)O"                       # functional groups
molprop isomers "CC(=O)NC1=CC=C(O)C=C1" --max-tautomers 10
molprop compare "CCO" "CCN"
molprop mcs "CC(=O)Oc1ccccc1C(=O)O" "CC(=O)Nc1ccc(O)cc1"
molprop alerts "Cc1ccc(cc1)/N=N/c2ccc(cc2)N(C)C" --catalog PAINS --catalog BRENK
molprop standardize "CC(=O)O.[Na]"
molprop admet "CC(=O)Oc1ccccc1C(=O)O"
molprop substructure "c1ccccc1" --project drug-x
molprop substructure "[CX3](=O)[OX2H1]" --candidates ./mols.txt

# One-click Markdown report
molprop report "CC(=O)NC1=CC=C(O)C=C1" -o acetaminophen.md
molprop report "CCO" --no-admet --no-descriptors    # lighter

# Library
molprop library save "CCO" --name Ethanol --tag solvent --tag polar --project drug-x
molprop library list --project drug-x --search etha
molprop library get 42
molprop library projects
molprop library delete 42       # prompts for confirmation
```

### Configuration

| Option       | Env var         | Default                  |
| ------------ | --------------- | ------------------------ |
| `--url`      | `MOLPROP_URL`   | `http://localhost:8000`  |
| `--timeout`  | —               | `60.0` seconds           |
| `--pretty/--compact` | —       | `--pretty`               |

Errors from the API are printed in red to stderr with a non-zero exit code,
suitable for shell pipelines:

```bash
molprop predict "INVALID" || echo "failed!"
```

---

## 4. End-to-end example

```bash
# 1. Save a batch of compounds
for smi in "CCO" "CCN" "c1ccccc1O"; do
  molprop library save "$smi" --project demo
done

# 2. Run a substructure search across the project
molprop substructure "c1ccccc1" --project demo

# 3. Compare the two closest hits
molprop compare "CCO" "c1ccccc1O"

# 4. Generate a Markdown report for each
molprop library list --project demo | jq -r '.[].smiles' | while read s; do
  molprop report "$s" -o "report_${s//[^a-zA-Z0-9]/_}.md"
done
```

---

## 5. Testing the SDK / CLI

```bash
# Storage + cheminformatics unit tests
pytest tests/test_storage.py tests/test_cheminformatics_features.py -v

# Full API integration tests (uses FastAPI TestClient — no server required)
pytest tests/test_api_integration.py -v
```
