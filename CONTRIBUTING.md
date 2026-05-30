# Contributing to Molecular Property Prediction System

We welcome contributions to the molprop project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip and venv (or conda)
- Git

### Setting Up Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jitesh523/Molecular-Property-Prediction-System.git
   cd Molecular-Property-Prediction-System
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   pip install pytest pytest-cov pytest-asyncio ruff black isort mypy flake8 bandit
   ```

### Project Structure

```
molprop/
├── src/molprop/              # Main source code
│   ├── config.py            # Configuration management
│   ├── exceptions.py        # Custom exceptions
│   ├── logger.py            # Logging setup
│   ├── cli.py               # Command-line interface
│   ├── client.py            # HTTP client
│   ├── data/                # Data ingestion and processing
│   ├── features/            # Feature extraction
│   ├── models/              # ML models
│   ├── serving/             # API and serving
│   ├── storage/             # Database and persistence
│   └── utils/               # Utilities and helpers
├── tests/                    # Test suite
├── notebooks/               # Jupyter notebooks
├── scripts/                 # Training and utility scripts
└── docs/                    # Documentation
```

## Development Workflow

### 1. Create a Branch

Create a feature or bugfix branch from `main`:

```bash
git checkout -b feature/my-feature
# or
git checkout -b bugfix/issue-description
```

Use descriptive branch names (e.g., `feature/cheminformatics-validation`, `bugfix/api-timeout`).

### 2. Make Changes

Make your changes to the codebase. Follow these guidelines:

#### Code Style

- **Line length:** 120 characters (configured in `.flake8` and `pyproject.toml`)
- **Formatter:** Black (run `black src/ tests/`)
- **Import sorting:** isort (run `isort src/ tests/`)
- **Type hints:** Add type hints to function signatures

#### Type Hints

Always add type hints to function signatures:

```python
def validate_smiles(smiles: str, allow_empty: bool = False) -> str:
    """Validate and canonicalize a SMILES string."""
    ...
```

#### Documentation

Add docstrings to all public functions and classes using Google style:

```python
def example_function(param1: str, param2: int) -> dict:
    """
    Brief description of what the function does.

    Longer description can go here if needed, explaining the algorithm,
    edge cases, or important notes.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        A dictionary containing the results.

    Raises:
        ValidationError: If param1 is invalid.
        ValueError: If param2 is negative.

    Examples:
        >>> result = example_function("test", 42)
        >>> result["status"]
        "success"
    """
```

#### Error Handling

Use custom exceptions from `molprop.exceptions`:

```python
from molprop.exceptions import ValidationError, InvalidSMILESError

def process_smiles(smiles: str) -> None:
    if not smiles:
        raise ValidationError("SMILES cannot be empty", field="smiles")
    
    if not is_valid_smiles(smiles):
        raise InvalidSMILESError(smiles, "Invalid structure")
```

### 3. Write Tests

Add tests for your changes:

```bash
pytest tests/test_your_feature.py -v
```

Test coverage should be maintained or improved. Run coverage report:

```bash
pytest tests/ --cov=molprop --cov-report=html
```

### 4. Run Linting and Type Checks

Before committing, run code quality tools:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/ --fix
flake8 src/ tests/

# Type checking
mypy src/ --ignore-missing-imports

# Security checks
bandit -r src/ -ll
```

### 5. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add SMILES validation with canonical conversion

- Implement comprehensive SMILES validation
- Add support for empty SMILES with flag
- Improve error messages with rdkit details
- Add 100% test coverage for validators
"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for code reorganization
- `perf:` for performance improvements
- `chore:` for maintenance

### 6. Push and Create Pull Request

```bash
git push origin feature/my-feature
```

Create a Pull Request (PR) on GitHub with:
- Clear title and description
- Reference to related issues (e.g., "Fixes #123")
- Summary of changes
- Testing instructions if applicable

## Testing Guidelines

### Writing Tests

- Use `pytest` as the test framework
- Place tests in the `tests/` directory with `test_` prefix
- Use descriptive test names: `test_validate_smiles_with_invalid_input`
- Group related tests in classes: `TestSMILESValidation`

Example test:

```python
import pytest
from molprop.exceptions import InvalidSMILESError
from molprop.utils.validators import validate_smiles

class TestSMILESValidation:
    def test_validate_smiles_with_valid_input(self):
        """Test that valid SMILES are accepted."""
        result = validate_smiles("CCO")
        assert result == "CCO"
    
    def test_validate_smiles_with_invalid_input(self):
        """Test that invalid SMILES raise an error."""
        with pytest.raises(InvalidSMILESError):
            validate_smiles("INVALID_SMILES")
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_validators.py

# Run specific test
pytest tests/test_validators.py::TestSMILESValidation::test_validate_smiles_with_valid_input

# Run with coverage
pytest tests/ --cov=molprop --cov-report=term-missing

# Run tests in parallel (faster)
pytest tests/ -n auto
```

## Documentation

When adding new features, update the relevant documentation:

- **Docstrings:** Add comprehensive docstrings to all functions and classes
- **README.md:** Update if the feature is user-facing
- **docs/:** Add new documentation files if needed
- **Examples:** Add examples in notebooks or example scripts

## Common Issues

### Import Errors

If you get import errors, ensure:
1. Package is installed in development mode: `pip install -e .`
2. Python path is correct (tests should have `pythonpath = ["src"]` in pytest config)
3. Virtual environment is activated

### Test Failures

If tests fail:
1. Check if dependencies are installed: `pip install -e .`
2. Run tests with verbose output: `pytest tests/ -vv`
3. Check if there are platform-specific issues (Windows vs macOS/Linux)

### Type Checking Errors

If mypy complains:
1. Add type hints to your functions
2. For third-party packages without stubs, add them to the `ignore_missing_imports` section in `pyproject.toml`
3. Use `# type: ignore` comments sparingly and only when necessary

## Review Process

Your PR will be reviewed by maintainers. They may:
- Request changes
- Ask for clarifications
- Suggest improvements
- Run additional checks

Address feedback promptly and keep the PR focused on a single feature or fix.

## Questions?

- Open an issue for bug reports or feature requests
- Check existing issues and discussions
- Read the project documentation in `docs/`

Thank you for contributing! 🎉
