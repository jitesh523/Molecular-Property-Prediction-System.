# Development Guide for molprop

This guide provides detailed information for developers working on the molprop codebase.

## Architecture Overview

The molprop system is organized into several layers:

```
┌─────────────────────────────────────────────────────────┐
│               CLI & REST API Layer                      │
│           (cli.py, serving/api.py)                      │
├─────────────────────────────────────────────────────────┤
│          Client & Configuration Layer                   │
│   (client.py, config.py, logger.py, exceptions.py)     │
├─────────────────────────────────────────────────────────┤
│            Feature Engineering Layer                    │
│  (data/, features/) - SMILES processing & descriptors  │
├─────────────────────────────────────────────────────────┤
│               Model Layer                               │
│  (models/) - GNNs, VAE, explanations, optimization      │
├─────────────────────────────────────────────────────────┤
│        Storage & Persistence Layer                      │
│    (serving/load_model.py, storage/library.py)          │
└─────────────────────────────────────────────────────────┘
```

## Module Breakdown

### Core Modules

#### `molprop/config.py`
Centralized configuration management using dataclasses.

**Key Classes:**
- `ModelConfig`: ML model settings (type, weights, dataset, task)
- `APIConfig`: FastAPI and server settings
- `CacheConfig`: Caching backend configuration
- `RateLimitConfig`: Rate limiting settings
- `DatabaseConfig`: Storage paths and database settings
- `Config`: Main composition of all configs

**Usage:**
```python
from molprop.config import get_config

config = get_config()
print(config.api.port)  # 8000
print(config.model.type)  # 'gcn'
```

#### `molprop/exceptions.py`
Custom exception hierarchy for domain-specific errors.

**Exception Types:**
- `MolpropError`: Base exception
- `ValidationError`: Input validation failures
- `InvalidSMILESError`: SMILES parsing failures
- `CheminformaticsError`: Chemical structure operations
- `ModelError`, `ModelLoadError`, `InferenceError`: Model operations
- `APIError`, `BadRequestError`, `NotFoundError`: HTTP errors
- `RateLimitError`, `StorageError`, `TimeoutError`: Operational errors

**Usage:**
```python
from molprop.exceptions import InvalidSMILESError, ValidationError

try:
    validate_smiles(invalid_smiles)
except InvalidSMILESError as e:
    print(f"Invalid SMILES: {e.smiles}")
    print(f"Error code: {e.code}")
```

#### `molprop/logger.py`
Structured logging configuration.

**Setup:**
```python
from molprop.logger import get_logger, setup_logging

# Automatic setup on import
log = get_logger(__name__)
log.info("Starting application")

# Manual setup with custom options
setup_logging(level="DEBUG", log_file="app.log", use_json=True)
```

#### `molprop/utils/validators.py`
Input validation utilities for common types.

**Validators:**
- `validate_smiles(smiles)`: Validate and canonicalize SMILES
- `validate_smiles_list(smiles_list)`: Validate multiple SMILES
- `validate_project_name(project)`: Alphanumeric project names
- `validate_tags(tags)`: Clean and deduplicate tags
- `validate_batch_size(batch_size)`: Ensure valid batch size

**Usage:**
```python
from molprop.utils.validators import validate_smiles, validate_batch_size

canonical = validate_smiles("CC(C)O")  # Returns canonicalized SMILES
batch_size = validate_batch_size(32)   # Returns 32 (validated)
```

#### `molprop/utils/decorators.py`
Reusable decorators for common patterns.

**Decorators:**
- `@timing`: Log execution time
- `@retry`: Retry on failure with backoff
- `@handle_errors`: Catch and log exceptions
- `@validate_input`: Validate function arguments

**Usage:**
```python
from molprop.utils.decorators import timing, retry, validate_input
from molprop.utils.validators import validate_smiles

@retry(max_attempts=3, delay_seconds=1.0)
@timing
def predict_smiles(smiles: str) -> dict:
    # Function is retried on failure and execution time is logged
    pass

@validate_input(smiles=validate_smiles, batch_size=lambda x: validate_batch_size(x))
def batch_predict(smiles: str, batch_size: int) -> dict:
    # Arguments are validated before function execution
    pass
```

### Feature Modules

#### `molprop/data/`
Data ingestion, standardization, and preprocessing.

**Key Files:**
- `processor.py`: Main data pipeline
- `standardize.py`: SMILES standardization (Lipinski, Veber, Ghose rules)
- `splits.py`: Train/test splitting strategies
- `ingest_chembl.py`, `ingest_pubchem.py`: External data sources
- `smiles_vocab.py`: SMILES tokenization vocabulary

#### `molprop/features/`
Chemical structure and molecular descriptor extraction.

**Key Modules:**
- `descriptors.py`: RDKit molecular descriptors
- `fingerprints.py`: Morgan, MACCS fingerprints
- `graphs.py`: PyTorch Geometric graph conversion
- `scaffolds.py`: Bemis-Murcko scaffold analysis
- `isomers.py`: Tautomer and stereoisomer enumeration
- `mcs.py`: Maximum Common Substructure
- `substructure.py`: Substructure matching
- `admet.py`: ADMET property prediction
- `functional_groups.py`: Functional group detection
- `conformers.py`: 3D conformer generation

#### `molprop/models/`
Machine learning models for predictions and explanations.

**Models:**
- `gnn_*.py`: Graph Neural Networks (GCN, GAT, GIN, MPNN)
- `transformer.py`: ChemBERTa transformer
- `vae.py`: Variational Autoencoder for generation
- `baselines.py`: Random Forest, XGBoost baselines
- `explain.py`, `explain_baselines.py`: Model interpretability
- `pareto.py`: Multi-objective optimization
- `optimization.py`: Latent space optimization

#### `molprop/serving/`
API and model serving layer.

**Key Files:**
- `api.py`: Main FastAPI application (30+ endpoints)
- `load_model.py`: Model loading and initialization
- `cache.py`: LRU TTL caching with JSON serialization
- `rate_limit.py`: Token bucket rate limiting
- `vector_db.py`: Vector search using Qdrant

#### `molprop/storage/`
Persistent storage and database operations.

**Key Files:**
- `library.py`: SQLite-backed compound library with CRUD operations

### CLI & Client

#### `molprop/cli.py`
Command-line interface with 19 subcommands.

**Usage:**
```bash
molprop predict "CCO"
molprop scaffold "CC(=O)Oc1ccccc1C(=O)O"
molprop library save "CCO" --project my-project --tag solvent
```

#### `molprop/client.py`
Python HTTP client for the REST API.

**Usage:**
```python
from molprop.client import MolpropClient

client = MolpropClient("http://localhost:8000")
result = client.predict("CCO")
print(result["prediction"])

# Batch operations
results = client.predict_batch(["CCO", "CCN", "CCCO"])

# Library operations
client.library_save("CCO", name="Ethanol", tags=["solvent"])
hits = client.substructure("c1ccccc1", project="my-project")
```

## Development Patterns

### Error Handling

Always use specific exception types:

```python
from molprop.exceptions import InvalidSMILESError, ValidationError, APIError

try:
    smiles = validate_smiles(user_input)
except ValidationError as e:
    logger.error(f"Validation failed for field '{e.field}': {e.message}")
    return APIError(f"Invalid {e.field}", status_code=400, details={"field": e.field})
except InvalidSMILESError as e:
    logger.error(f"Invalid SMILES '{e.smiles}': {e.message}")
    return APIError("Invalid SMILES", status_code=400, details={"smiles": e.smiles})
```

### Type Hints

Always add type hints:

```python
from typing import Optional, List
from pathlib import Path

def process_molecules(
    smiles_list: List[str],
    output_path: Optional[Path] = None,
    max_size: int = 1000,
) -> dict:
    """
    Process a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings to process.
        output_path: Optional path to save results.
        max_size: Maximum number of molecules to process.
        
    Returns:
        Dictionary with processing results.
        
    Raises:
        ValidationError: If inputs are invalid.
    """
    ...
```

### Logging

Use the logger module for all logging:

```python
from molprop.logger import get_logger

log = get_logger(__name__)

log.debug("Starting operation X")
log.info("Successfully processed Y items")
log.warning("Deprecation warning: use new_function instead")
log.error("Failed to process Z", exc_info=True)
```

### Configuration Access

Access configuration through the config module:

```python
from molprop.config import get_config

config = get_config()
model_type = config.model.type
api_port = config.api.port
cache_size = config.cache.max_size
```

## Testing

### Test Structure

Tests should follow this structure:

```python
import pytest
from molprop.exceptions import InvalidSMILESError
from molprop.utils.validators import validate_smiles

class TestSMILESValidation:
    """Test suite for SMILES validation."""
    
    def test_validate_smiles_with_valid_input(self):
        """Valid SMILES should be canonicalized."""
        result = validate_smiles("CCO")
        assert result == "CCO"
    
    def test_validate_smiles_with_alternative_notation(self):
        """Different SMILES notations should be canonicalized to same result."""
        result = validate_smiles("OCC")  # Alternative notation for ethanol
        assert result == "CCO"
    
    def test_validate_smiles_with_invalid_input(self):
        """Invalid SMILES should raise InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError):
            validate_smiles("INVALID_SMILES")
    
    def test_validate_smiles_with_empty_input(self):
        """Empty SMILES should raise ValidationError."""
        from molprop.exceptions import ValidationError
        with pytest.raises(ValidationError):
            validate_smiles("")
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=molprop --cov-report=html

# Run specific test class
pytest tests/test_validators.py::TestSMILESValidation

# Run with verbose output
pytest tests/ -vv

# Run in parallel
pytest tests/ -n auto
```

## Performance Considerations

### Threading and Async

- FastAPI endpoints use `ThreadPoolExecutor` for CPU-bound inference to avoid blocking async event loop
- Use `asyncio.run_in_executor()` for blocking operations in async context

### Caching

- Configure cache size and TTL in `config.py`
- Cache is LRU-based with TTL expiration
- Use `@cached_json` decorator for endpoint caching

### Batching

- Batch endpoints group multiple SMILES for efficient processing
- Use `validate_batch_size()` to ensure batch sizes are within limits

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Troubleshooting

### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Check Python path is correct
python -c "import molprop; print(molprop.__file__)"
```

### Test Failures
```bash
# Run with verbose output
pytest tests/ -vv

# Run specific test with output
pytest tests/test_file.py::TestClass::test_method -vv -s
```

### Type Checking Errors
```bash
# Run mypy with ignore missing imports
mypy src/ --ignore-missing-imports

# Check specific file
mypy src/molprop/config.py
```

## Resources

- **RDKit Documentation**: https://www.rdkit.org/docs/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Pydantic**: https://docs.pydantic.dev/
