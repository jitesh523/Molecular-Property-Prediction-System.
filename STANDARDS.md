# Code Quality and Documentation Standards

This document outlines the code quality standards, documentation requirements, and best practices for the molprop project.

## Code Quality Standards

### Type Hints

All public functions and methods must have type hints:

```python
from typing import Optional, List, Dict, Tuple

def predict_batch(
    smiles_list: List[str],
    model_type: Optional[str] = None,
) -> Dict[str, any]:
    """Predict properties for multiple SMILES."""
    pass
```

### Docstring Format

Use Google-style docstrings for all public APIs:

```python
def example_function(
    param1: str,
    param2: int,
    param3: Optional[str] = None,
) -> Dict[str, any]:
    """
    Brief one-line description of what this function does.

    Longer description explaining the algorithm, any important notes,
    or usage patterns. Can span multiple paragraphs.

    Args:
        param1: Description of param1 and what it should contain.
        param2: Description of param2 with valid ranges or constraints.
        param3: Optional parameter description. Defaults to None.

    Returns:
        Dictionary containing:
            - 'result': The main result
            - 'metadata': Additional metadata

    Raises:
        ValidationError: If param1 is invalid.
        ValueError: If param2 is negative.
        TimeoutError: If operation exceeds timeout.

    Examples:
        Basic usage:
        >>> result = example_function("test", 42)
        >>> result['result']
        'processed'

        With optional parameter:
        >>> result = example_function("test", 42, param3="custom")
    """
    pass
```

### Imports

Organize imports according to PEP 8:

```python
# Standard library
import json
import logging
from pathlib import Path
from typing import Optional, List

# Third-party libraries
import numpy as np
import pandas as pd
from rdkit import Chem
from pydantic import BaseModel

# Local imports
from molprop.config import get_config
from molprop.exceptions import ValidationError
from molprop.logger import get_logger
from molprop.utils.validators import validate_smiles
```

### Line Length

Maximum line length is 120 characters. Configured in `.flake8` and `pyproject.toml`.

### Naming Conventions

- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE = 1000`)
- **Functions/Variables**: `lower_snake_case` (e.g., `process_molecules`, `result_list`)
- **Classes**: `PascalCase` (e.g., `MoleculeProcessor`, `Config`)
- **Private/Internal**: Prefix with `_` (e.g., `_internal_helper`, `_INTERNAL_CONSTANT`)

### Exception Handling

Never use bare `except`:

```python
# ❌ DON'T
try:
    some_operation()
except:
    pass

# ✅ DO
try:
    some_operation()
except ValidationError as e:
    logger.error(f"Validation failed: {e.message}")
except (TimeoutError, RuntimeError) as e:
    logger.error(f"Operation failed: {str(e)}")
except Exception as e:  # Catch-all only if absolutely necessary
    logger.exception("Unexpected error occurred")
    raise
```

### Logging

Always use the logger module, not print statements:

```python
from molprop.logger import get_logger

log = get_logger(__name__)

# ❌ DON'T
print("Processing started")

# ✅ DO
log.info("Processing started")
log.debug("Processing details: %s", details)
log.warning("Deprecated function used")
log.error("Processing failed", exc_info=True)
```

## Documentation Requirements

### README.md

The README should include:
- Project description and use cases
- Quick start guide
- Installation instructions
- API overview
- Example usage
- Contributing guidelines link
- License information

### CONTRIBUTING.md

Contribution guidelines should cover:
- Code of conduct
- Development setup
- Workflow for making changes
- Code style requirements
- Testing guidelines
- Commit message conventions
- Review process

### DEVELOPMENT.md

Development documentation should include:
- Architecture overview
- Module descriptions
- Pattern examples
- Configuration guide
- Testing procedures
- Troubleshooting

### Module Docstrings

Every module should start with a docstring:

```python
"""
molprop.features.validators — Input validation utilities.

Provides validators for common types like SMILES strings, batch sizes,
and project names with helpful error messages.

Example:
    >>> from molprop.features.validators import validate_smiles
    >>> canonical = validate_smiles("CCO")
"""
```

### Class Docstrings

```python
class MoleculeProcessor:
    """
    High-level molecule processing pipeline.

    Handles standardization, descriptor computation, and feature extraction
    for molecular structures.

    Attributes:
        max_batch_size: Maximum molecules to process in one batch.
        standardizer: SMILES standardizer instance.
        descriptor_calculator: Descriptor computation engine.

    Example:
        processor = MoleculeProcessor(max_batch_size=100)
        results = processor.process("CCO")
    """
```

## Testing Standards

### Test Organization

```
tests/
├── test_validators.py        # Unit tests for validators
├── test_models.py            # Model tests
├── test_api_integration.py   # Integration tests
└── conftest.py              # Pytest configuration
```

### Test Naming

- File: `test_<module>.py`
- Class: `Test<Feature>` (e.g., `TestSMILESValidation`)
- Method: `test_<scenario>` (e.g., `test_validate_smiles_with_invalid_input`)

### Pytest Fixtures

Use fixtures for common setup:

```python
import pytest

@pytest.fixture
def client():
    """Create an API client for testing."""
    from molprop.client import MolpropClient
    return MolpropClient("http://localhost:8000")

@pytest.fixture
def sample_smiles():
    """Provide sample SMILES for testing."""
    return ["CCO", "CC(C)O", "c1ccccc1"]

def test_batch_predict(client, sample_smiles):
    """Test batch prediction."""
    result = client.predict_batch(sample_smiles)
    assert len(result["predictions"]) == len(sample_smiles)
```

### Test Coverage

Aim for at least 80% code coverage:

```bash
# Run tests with coverage report
pytest tests/ --cov=molprop --cov-report=html
```

## Security Standards

### Input Validation

Always validate user inputs:

```python
from molprop.utils.validators import validate_smiles
from molprop.exceptions import ValidationError

@app.post("/predict")
def predict(smiles: str):
    try:
        validated_smiles = validate_smiles(smiles)
    except ValidationError as e:
        return {"error": str(e)}, 400
    # ... process
```

### Dependency Security

Regularly check for vulnerabilities:

```bash
# Check for known vulnerabilities
safety check

# Audit pip packages
pip-audit

# Bandit for code security issues
bandit -r src/ -ll
```

### No Hardcoded Secrets

Never hardcode credentials, API keys, or sensitive data:

```python
# ❌ DON'T
API_KEY = "sk-1234567890abcdef"

# ✅ DO
import os
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ConfigurationError("API_KEY environment variable not set")
```

## Performance Standards

### Efficient Algorithms

Choose appropriate algorithms for the task:

```python
# ❌ DON'T - O(n²) for simple lookup
for item in large_list:
    if item == target:
        found = True
        break

# ✅ DO - O(1) lookup
if target in set(large_list):
    found = True
```

### Profiling

Use profiling to identify bottlenecks:

```python
from molprop.utils.decorators import timing

@timing
def expensive_operation():
    """This will log execution time."""
    pass
```

### Memory Efficiency

Use generators for large datasets:

```python
# ❌ DON'T - Loads entire dataset into memory
def process_molecules(smiles_list):
    results = []
    for smiles in smiles_list:
        results.append(process(smiles))
    return results

# ✅ DO - Generator for memory efficiency
def process_molecules(smiles_list):
    for smiles in smiles_list:
        yield process(smiles)
```

## Continuous Integration Standards

The project uses GitHub Actions for CI/CD. All PRs must pass:

1. **Linting**: ruff, flake8
2. **Formatting**: black, isort
3. **Type Checking**: mypy
4. **Security**: bandit, safety
5. **Tests**: pytest with coverage
6. **Build**: Package builds successfully

## Version Management

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

Update version in `pyproject.toml`:
```toml
[project]
version = "2.45.1"
```

## Changelog

Maintain a CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [2.45.1] - 2024-01-15

### Added
- New feature description

### Fixed
- Bug fix description

### Changed
- Breaking change description
```

## Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines (black, isort, flake8)
- [ ] Type hints added for all public functions
- [ ] Docstrings updated for new/modified functions
- [ ] Tests written and all tests pass
- [ ] Coverage maintained or improved
- [ ] No new warnings or errors
- [ ] Commit messages follow conventions
- [ ] CHANGELOG.md updated if needed
- [ ] No hardcoded credentials or sensitive data
- [ ] All linting, type checking, and security checks pass
