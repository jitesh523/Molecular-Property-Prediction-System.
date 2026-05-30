"""
molprop — Production-grade molecular property prediction platform.

A comprehensive ML engineering system for predicting molecular properties
from chemical structures, with GNNs, cheminformatics features, and a
production REST API.

Quick Start:
    from molprop.client import MolpropClient
    from molprop.config import get_config
    from molprop.logger import get_logger

    # Get configuration
    config = get_config()

    # Create API client
    client = MolpropClient("http://localhost:8000")
    result = client.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin

    # Use logging
    log = get_logger(__name__)
    log.info("Prediction result: %s", result)
"""

__version__ = "2.45.0"

# Core configuration and logging
from molprop.config import APIConfig, CacheConfig, Config, DatabaseConfig, ModelConfig, RateLimitConfig, get_config
from molprop.exceptions import (
    APIError,
    BadRequestError,
    CheminformaticsError,
    ConfigurationError,
    InferenceError,
    InvalidSMILESError,
    ModelError,
    ModelLoadError,
    MolpropError,
    NotFoundError,
    RateLimitError,
    ServerError,
    StorageError,
    TimeoutError,
    ValidationError,
    get_exception_for_status,
)
from molprop.logger import get_logger, setup_logging

# Client and CLI
from molprop.client import MolpropAPIError, MolpropClient

# Utilities
from molprop.utils.decorators import handle_errors, retry, timing, validate_input
from molprop.utils.validators import (
    validate_batch_size,
    validate_project_name,
    validate_smiles,
    validate_smiles_list,
    validate_tags,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "APIConfig",
    "CacheConfig",
    "Config",
    "DatabaseConfig",
    "ModelConfig",
    "RateLimitConfig",
    "get_config",
    # Exceptions
    "APIError",
    "BadRequestError",
    "CheminformaticsError",
    "ConfigurationError",
    "InferenceError",
    "InvalidSMILESError",
    "ModelError",
    "ModelLoadError",
    "MolpropError",
    "NotFoundError",
    "RateLimitError",
    "ServerError",
    "StorageError",
    "TimeoutError",
    "ValidationError",
    "get_exception_for_status",
    # Logging
    "get_logger",
    "setup_logging",
    # Client
    "MolpropAPIError",
    "MolpropClient",
    # Utilities
    "handle_errors",
    "retry",
    "timing",
    "validate_input",
    "validate_batch_size",
    "validate_project_name",
    "validate_smiles",
    "validate_smiles_list",
    "validate_tags",
]
