# API Documentation

Complete reference for the molprop REST API with 30+ endpoints for molecular property prediction, generation, and analysis.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API doesn't require authentication. In production, add API keys:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/health
```

## Response Format

All responses are JSON:

### Success Response

```json
{
  "prediction": 0.85,
  "confidence": 0.92,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### Error Response

```json
{
  "detail": "Invalid SMILES: Invalid structure",
  "error_code": "INVALID_SMILES",
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

## System Endpoints

### Health Check

Check if API is running and responsive.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "version": "2.45.0"
}
```

### Version

Get API and package versions.

**Request:**
```bash
curl http://localhost:8000/version
```

**Response:**
```json
{
  "api_version": "2.45.0",
  "python_version": "3.11.0",
  "torch_version": "2.2.1",
  "rdkit_version": "2023.09.5"
}
```

### Model Info

Get information about the loaded ML model.

**Request:**
```bash
curl http://localhost:8000/model/info
```

**Response:**
```json
{
  "model_type": "gcn",
  "dataset": "bbbp",
  "task": "classification",
  "input_features": 39,
  "hidden_dim": 128,
  "num_classes": 2
}
```

### Metrics

Get performance metrics.

**Request:**
```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
  "total_requests": 1234,
  "predictions_served": 5678,
  "avg_response_time_ms": 45.2,
  "cache_hit_rate": 0.65,
  "errors": 2
}
```

### Cache Stats

Get cache statistics.

**Request:**
```bash
curl http://localhost:8000/cache/stats
```

**Response:**
```json
{
  "size": 256,
  "max_size": 512,
  "hit_count": 1200,
  "miss_count": 300,
  "hit_rate": 0.80
}
```

### Clear Cache

Clear the entire cache.

**Request:**
```bash
curl -X POST http://localhost:8000/cache/clear
```

**Response:**
```json
{
  "status": "cache cleared",
  "items_removed": 256
}
```

## Prediction Endpoints

### Single Prediction

Predict molecular property for a single SMILES.

**Endpoint:** `POST /predict`

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'
```

**Request Body:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O"
}
```

**Response:**
```json
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "prediction": 0.85,
  "confidence": [0.15, 0.85],
  "processing_time_ms": 25
}
```

### Batch Prediction

Predict properties for multiple SMILES in one request.

**Endpoint:** `POST /predict/batch`

**Request:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "smiles_list": ["CCO", "CCN", "c1ccccc1"],
    "batch_size": 32
  }'
```

**Request Body:**
```json
{
  "smiles_list": ["CCO", "CCN", "c1ccccc1"],
  "batch_size": 32
}
```

**Response:**
```json
{
  "predictions": [
    {"smiles": "CCO", "prediction": 0.25},
    {"smiles": "CCN", "prediction": 0.30},
    {"smiles": "c1ccccc1", "prediction": 0.95}
  ],
  "total_time_ms": 50,
  "processed": 3
}
```

## Molecular Analysis Endpoints

### Scaffold Analysis

Analyze Bemis-Murcko scaffolds and synthetic accessibility.

**Endpoint:** `POST /scaffold`

**Request:**
```bash
curl -X POST http://localhost:8000/scaffold \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)NC1=CC=C(O)C=C1"}'
```

**Response:**
```json
{
  "smiles": "CC(=O)NC1=CC=C(O)C=C1",
  "murcko_smiles": "c1ccc(O)cc1",
  "sa_score": 2.5,
  "sa_category": "easy",
  "rings": 1,
  "aromatic_rings": 1
}
```

### Functional Groups

Identify functional groups in molecule.

**Endpoint:** `POST /functional_groups`

**Request:**
```bash
curl -X POST http://localhost:8000/functional_groups \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)NC1=CC=C(O)C=C1"}'
```

**Response:**
```json
{
  "smiles": "CC(=O)NC1=CC=C(O)C=C1",
  "functional_groups": [
    {"name": "amide", "count": 1, "matches": [1]},
    {"name": "phenol", "count": 1, "matches": [5]},
    {"name": "aromatic_ring", "count": 1}
  ]
}
```

### Isomers

Enumerate stereoisomers and tautomers.

**Endpoint:** `POST /isomers`

**Request:**
```bash
curl -X POST http://localhost:8000/isomers \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(C)O", "max_results": 10}'
```

**Response:**
```json
{
  "input_smiles": "CC(C)O",
  "stereoisomers": ["CC(C)O", "C(C)(C)O"],
  "tautomers": ["CC(C)O", "CC(C)O"],
  "unique": ["CC(C)O"],
  "count": 1
}
```

### Substructure Search

Search for substructure matches.

**Endpoint:** `POST /substructure`

**Request:**
```bash
curl -X POST http://localhost:8000/substructure \
  -H "Content-Type: application/json" \
  -d '{
    "query_smiles": "c1ccccc1",
    "project": "my_library",
    "limit": 50
  }'
```

**Response:**
```json
{
  "query_smiles": "c1ccccc1",
  "matches": [
    {
      "compound_id": "1",
      "smiles": "c1ccccc1O",
      "name": "Phenol",
      "n_matches": 1
    }
  ],
  "total_matches": 45,
  "limit": 50
}
```

### Similarity Search

Find similar molecules using fingerprint Tanimoto similarity.

**Endpoint:** `POST /search/similar`

**Request:**
```bash
curl -X POST http://localhost:8000/search/similar \
  -H "Content-Type: application/json" \
  -d '{
    "query_smiles": "CCO",
    "threshold": 0.75,
    "fingerprint_type": "morgan",
    "limit": 100
  }'
```

**Response:**
```json
{
  "query_smiles": "CCO",
  "similar": [
    {
      "compound_id": "2",
      "smiles": "CCCO",
      "similarity": 0.89,
      "name": "1-Propanol"
    }
  ],
  "total_found": 25,
  "threshold": 0.75
}
```

### Maximum Common Substructure

Find MCS between two molecules.

**Endpoint:** `POST /mcs`

**Request:**
```bash
curl -X POST http://localhost:8000/mcs \
  -H "Content-Type: application/json" \
  -d '{
    "smiles1": "c1ccccc1O",
    "smiles2": "c1ccccc1N"
  }'
```

**Response:**
```json
{
  "smiles1": "c1ccccc1O",
  "smiles2": "c1ccccc1N",
  "mcs_smiles": "c1ccccc1",
  "mcs_size": 6,
  "similarity": 0.95
}
```

### Compare Molecules

Compare multiple molecular properties and similarity.

**Endpoint:** `POST /compare`

**Request:**
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "smiles_list": ["CCO", "CCN", "CCCO"]
  }'
```

**Response:**
```json
{
  "molecules": [
    {
      "smiles": "CCO",
      "mw": 46.04,
      "logp": -0.76,
      "hbd": 1,
      "hba": 1
    }
  ],
  "similarities": [[1.0, 0.88, 0.92]]
}
```

### Structure Alerts

Identify PAINS, Brenk, and NIH alerts.

**Endpoint:** `POST /alerts`

**Request:**
```bash
curl -X POST http://localhost:8000/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "Cc1ccc(cc1)/N=N/c2ccc(cc2)N(C)C",
    "catalogs": ["PAINS", "BRENK", "NIH"]
  }'
```

**Response:**
```json
{
  "smiles": "Cc1ccc(cc1)/N=N/c2ccc(cc2)N(C)C",
  "pains_alerts": ["azo_group"],
  "brenk_alerts": ["tertiary_amine"],
  "nih_alerts": []
}
```

## Generation Endpoints

### Generate Molecules

Generate new molecules using latent space optimization.

**Endpoint:** `POST /generate`

**Request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "num_molecules": 10,
    "seed": 42
  }'
```

**Response:**
```json
{
  "generated": [
    {"smiles": "CCO", "logp": 0.5, "mw": 100},
    {"smiles": "CCN", "logp": 0.3, "mw": 110}
  ],
  "count": 10
}
```

### Constrained Generation

Generate molecules with property constraints.

**Endpoint:** `POST /generate/smart`

**Request:**
```bash
curl -X POST http://localhost:8000/generate/smart \
  -H "Content-Type: application/json" \
  -d '{
    "num_molecules": 5,
    "mw_range": [200, 400],
    "logp_range": [0, 3],
    "seed": 42
  }'
```

**Response:**
```json
{
  "generated": [
    {"smiles": "CC(C)NC(=O)c1ccccc1", "properties": {...}}
  ],
  "constraints": {"mw_range": [200, 400], "logp_range": [0, 3]},
  "count": 5
}
```

## Library Management Endpoints

### Save Molecule to Library

Add molecule to persistent library.

**Endpoint:** `POST /library`

**Request:**
```bash
curl -X POST http://localhost:8000/library \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CCO",
    "name": "Ethanol",
    "project": "solvents",
    "tags": ["polar", "small"]
  }'
```

**Response:**
```json
{
  "id": "mol_123",
  "smiles": "CCO",
  "name": "Ethanol",
  "project": "solvents",
  "tags": ["polar", "small"],
  "created_at": "2024-01-15T10:30:45Z"
}
```

### List Library

Get molecules from library with filtering.

**Endpoint:** `GET /library`

**Request:**
```bash
curl "http://localhost:8000/library?project=solvents&tags=polar&limit=20"
```

**Response:**
```json
{
  "molecules": [
    {"id": "mol_123", "smiles": "CCO", "name": "Ethanol"}
  ],
  "total": 42,
  "limit": 20
}
```

### Export Library

Export library as CSV.

**Endpoint:** `GET /library/export/csv`

**Request:**
```bash
curl "http://localhost:8000/library/export/csv?project=solvents" > library.csv
```

### Import Library

Import molecules from CSV.

**Endpoint:** `POST /library/import`

**Request:**
```bash
curl -X POST http://localhost:8000/library/import \
  -F "file=@molecules.csv" \
  -F "project=my_project"
```

## Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `INVALID_SMILES` | 400 | SMILES string is invalid |
| `VALIDATION_ERROR` | 400 | Input validation failed |
| `NOT_FOUND` | 404 | Requested resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `TIMEOUT` | 504 | Operation exceeded timeout |
| `INTERNAL_SERVER_ERROR` | 500 | Server error |

## Rate Limiting

Default: 120 requests per 60 seconds per client.

**Response headers:**
```
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 119
X-RateLimit-Reset: 1234567890
```

## Examples

### Python Client

```python
from molprop.client import MolpropClient

client = MolpropClient("http://localhost:8000")

# Single prediction
result = client.predict("CCO")
print(f"Prediction: {result['prediction']}")

# Batch prediction
results = client.predict_batch(["CCO", "CCN", "CCCO"])

# Scaffold analysis
scaffold = client.scaffold("CC(=O)Oc1ccccc1C(=O)O")
print(f"SA Score: {scaffold['sa_score']}")

# Save to library
compound = client.library_save("CCO", name="Ethanol", tags=["solvent"])
print(f"Saved: {compound['id']}")
```

### cURL

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"smiles_list": ["CCO", "CCN", "CCCO"]}'

# Get health status
curl http://localhost:8000/health | python -m json.tool
```

### JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

async function predict(smiles) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles })
  });
  return response.json();
}

predict('CCO').then(console.log);
```
