# Quick Start Guide

Get started with molprop in 5 minutes.

## Installation

### Option 1: pip (Recommended for Users)

```bash
pip install molprop
```

### Option 2: From Source (for Development)

```bash
git clone https://github.com/jitesh523/Molecular-Property-Prediction-System.git
cd Molecular-Property-Prediction-System
pip install -e .
```

### Option 3: Docker

```bash
docker run -p 8000:8000 jitesh523/molprop:latest
```

## 1. Start the API Server

```bash
molprop-serve
# or
python -m uvicorn molprop.serving.api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 2. Predict Molecular Properties

### Using Python Client

```python
from molprop.client import MolpropClient

# Create client
client = MolpropClient("http://localhost:8000")

# Single prediction (Aspirin)
result = client.predict("CC(=O)Oc1ccccc1C(=O)O")
print(f"Prediction: {result['prediction']}")  # e.g., 0.85

# Batch prediction
smiles_list = ["CCO", "CCN", "c1ccccc1"]
results = client.predict_batch(smiles_list)
for r in results["predictions"]:
    print(f"{r['smiles']}: {r['prediction']}")
```

### Using CLI

```bash
# Single prediction
molprop predict "CC(=O)Oc1ccccc1C(=O)O"

# Batch prediction from file
molprop predict-file molecules.smi

# Format: SMILES (one per line)
```

### Using HTTP API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'
```

## 3. Analyze Molecular Properties

### Get Scaffold Information

```python
# Python
scaffold = client.scaffold("CC(=O)Oc1ccccc1C(=O)O")
print(f"SA Score (Synthetic Accessibility): {scaffold['sa_score']}")
print(f"Murcko Scaffold: {scaffold['murcko_smiles']}")
```

```bash
# CLI
molprop scaffold "CC(=O)Oc1ccccc1C(=O)O"
```

### Compare Molecules

```python
result = client.compare(["CCO", "CCN", "CCCO"])
print(result)  # Shows molecular descriptors and similarities
```

```bash
molprop compare "CCO" "CCN"
```

### Find Structural Alerts

```python
alerts = client.alerts("Cc1ccc(cc1)/N=N/c2ccc(cc2)N(C)C")
print(alerts)  # Shows PAINS, Brenk, NIH alerts
```

```bash
molprop alerts "Cc1ccc(cc1)/N=N/c2ccc(cc2)N(C)C" --catalog PAINS --catalog BRENK
```

## 4. Search Similar Molecules

### Similarity Search

```python
# Find molecules similar to ethanol
results = client.search_similar("CCO", threshold=0.75)
for r in results["similar"]:
    print(f"{r['smiles']}: {r['similarity']}")
```

### Substructure Search

```python
# Find all molecules with benzene ring
results = client.substructure("c1ccccc1")
for r in results["matches"]:
    print(f"{r['smiles']}")
```

## 5. Save and Manage Molecules

### Save Molecules to Library

```python
# Save single molecule
compound = client.library_save(
    "CCO",
    name="Ethanol",
    project="solvents",
    tags=["polar", "small"]
)
print(f"Saved as: {compound['id']}")

# Save multiple molecules
for smiles, name in [("CCO", "Ethanol"), ("CCCO", "Propanol")]:
    client.library_save(smiles, name=name, project="solvents")
```

```bash
# CLI
molprop library save "CCO" --name "Ethanol" --project "solvents" --tag "polar"
molprop library save "CCCO" --name "Propanol" --project "solvents" --tag "polar"
```

### List and Query Library

```python
# List all compounds in project
compounds = client.library_list(project="solvents")
for c in compounds["molecules"]:
    print(f"{c['name']}: {c['smiles']}")

# Search with tags
compounds = client.library_list(project="solvents", tags=["polar"])
```

```bash
# CLI
molprop library list --project "solvents"
molprop library search "benzene" --project "drug-x"
```

### Export/Import Library

```python
# Export to CSV
csv_data = client.library_export(project="solvents")

# Import from CSV
client.library_import("molecules.csv", project="my-project")
```

```bash
# CLI
molprop library export --project "solvents" -o solvents.csv
molprop library import molecules.csv --project "my-project"
```

## 6. Generate Molecules

### Generate Random Molecules

```python
# Generate 10 random valid molecules
results = client.generate(num_molecules=10, seed=42)
for mol in results["generated"]:
    print(mol["smiles"])
```

### Constrained Generation

```python
# Generate molecules with property constraints
results = client.generate_constrained(
    num_molecules=5,
    mw_range=[200, 400],
    logp_range=[0, 3],
    qed_range=[0.5, 1.0]
)
```

## 7. Create Reports

### Generate Markdown Report

```python
# Create detailed analysis report
report = client.report("CC(=O)Oc1ccccc1C(=O)O")
with open("aspirin_report.md", "w") as f:
    f.write(report)
```

```bash
# CLI
molprop report "CC(=O)Oc1ccccc1C(=O)O" -o aspirin.md
```

## Common Use Cases

### 1. Screen a Compound Library

```python
from molprop.client import MolpropClient

client = MolpropClient("http://localhost:8000")

# Read SMILES file
with open("compounds.smi") as f:
    smiles_list = [line.strip() for line in f]

# Batch predict
results = client.predict_batch(smiles_list)

# Filter by prediction
active = [r for r in results["predictions"] if r["prediction"] > 0.7]
print(f"Found {len(active)} active compounds")
```

### 2. Find Lead Compounds from Public Databases

```python
# Search similar to a reference compound
reference = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
similar = client.search_similar(reference, threshold=0.8)

# Analyze each similar compound
for mol in similar["similar"]:
    pred = client.predict(mol["smiles"])
    print(f"{mol['name']}: pred={pred['prediction']}, sim={mol['similarity']}")
```

### 3. Optimize Lead Compound

```python
# Start with a hit compound
hit_smiles = "CCO"

# Generate analogs with constraints
analogs = client.generate_constrained(
    num_molecules=100,
    mw_range=[100, 200],
    logp_range=[0, 2],
    qed_range=[0.6, 1.0]
)

# Predict on all analogs
results = client.predict_batch([m["smiles"] for m in analogs["generated"]])

# Sort by prediction
top_compounds = sorted(results["predictions"], key=lambda x: x["prediction"], reverse=True)[:10]
```

### 4. Build Project Library

```python
# Create a project library with your compounds
compounds_data = [
    {"smiles": "CCO", "name": "Ethanol", "tags": ["solvent", "small"]},
    {"smiles": "CCCO", "name": "Propanol", "tags": ["solvent", "small"]},
    {"smiles": "c1ccccc1", "name": "Benzene", "tags": ["aromatic"]},
]

for comp in compounds_data:
    client.library_save(
        comp["smiles"],
        name=comp["name"],
        project="my_project",
        tags=comp["tags"]
    )

# Later, query the library
library = client.library_list(project="my_project", tags=["solvent"])
```

## Configuration

### Environment Variables

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# Model Configuration
export MODEL_TYPE=gcn
export MODEL_DATASET=bbbp
export MODEL_TASK=classification

# Cache Configuration
export CACHE_ENABLED=true
export CACHE_MAX_SIZE=512
export CACHE_TTL_SECONDS=600

# Logging
export LOG_LEVEL=INFO
```

### Configuration File

Create `.molprop.yaml`:

```yaml
api:
  host: 0.0.0.0
  port: 8000
  workers: 4

model:
  type: gcn
  dataset: bbbp
  task: classification

cache:
  enabled: true
  max_size: 512
  ttl_seconds: 600

logging:
  level: INFO
  format: text  # or 'json'
```

## Next Steps

- Read the [API Documentation](API.md) for detailed endpoint reference
- Check out [examples/](../examples/) for more code samples
- See [notebooks/](../notebooks/) for interactive tutorials
- Review [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute
- Visit [docs/CLI_AND_SDK.md](CLI_AND_SDK.md) for CLI details

## Troubleshooting

### ImportError: No module named 'molprop'

**Solution:** Install the package:
```bash
pip install -e .
```

### Connection refused on localhost:8000

**Solution:** Start the API server first:
```bash
molprop-serve
# or manually
python -m uvicorn molprop.serving.api:app --port 8000
```

### Invalid SMILES error

**Solution:** Use valid SMILES notation. Test with examples:
```python
valid_smiles = [
    "CCO",               # Ethanol
    "c1ccccc1",          # Benzene
    "CC(C)O",            # Isopropanol
]
```

### Out of memory errors

**Solution:** Use batch predictions with smaller batch sizes:
```python
batch_size = 32  # Reduce if needed
for i in range(0, len(smiles_list), batch_size):
    batch = smiles_list[i:i+batch_size]
    results = client.predict_batch(batch)
```

## Getting Help

- **Documentation:** Check [docs/](../docs/)
- **Examples:** Browse [examples/](../examples/)
- **Issues:** Open an issue on [GitHub](https://github.com/jitesh523/Molecular-Property-Prediction-System)
- **Discussions:** Ask questions in GitHub Discussions

## License

MIT License - See [LICENSE](../LICENSE) for details
