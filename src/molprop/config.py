"""
Configuration management for molprop.

Centralizes environment variables, defaults, and application settings
with typed access and validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Model loading and inference configuration."""

    type: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "gcn"))
    weights_path: str = field(default_factory=lambda: os.getenv("MODEL_WEIGHTS", "best_model_gcn_bbbp.pt"))
    dataset: str = field(default_factory=lambda: os.getenv("MODEL_DATASET", "bbbp"))
    task: str = field(default_factory=lambda: os.getenv("MODEL_TASK", "classification"))
    target_col: str = field(default_factory=lambda: os.getenv("MODEL_TARGET_COL", ""))
    hidden_dim: int = 128
    atom_feature_dim: int = 39  # Standard PyG atom feature dimension

    def __post_init__(self) -> None:
        """Validate model configuration."""
        valid_types = {"gcn", "gat", "gin", "mpnn", "multitask"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid model type '{self.type}'. Must be one of {valid_types}")

        valid_tasks = {"classification", "regression", "multitask"}
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task '{self.task}'. Must be one of {valid_tasks}")


@dataclass
class APIConfig:
    """REST API configuration."""

    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))
    debug: bool = field(default_factory=lambda: os.getenv("API_DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    inference_timeout: float = 60.0
    batch_size: int = 32


@dataclass
class CacheConfig:
    """Caching configuration."""

    enabled: bool = field(default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true")
    max_size: int = field(default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE", "512")))
    ttl_seconds: float = field(default_factory=lambda: float(os.getenv("CACHE_TTL_SECONDS", "600")))
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_URL"))


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    enabled: bool = field(default_factory=lambda: os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true")
    capacity: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_CAPACITY", "120")))
    window_seconds: float = field(default_factory=lambda: float(os.getenv("RATE_LIMIT_WINDOW", "60")))


@dataclass
class DatabaseConfig:
    """Database/storage configuration."""

    library_db_path: Path = field(
        default_factory=lambda: Path(os.getenv("LIBRARY_DB_PATH", ".molprop/library.db"))
    )
    vector_db_enabled: bool = field(
        default_factory=lambda: os.getenv("VECTOR_DB_ENABLED", "true").lower() == "true"
    )
    vector_db_path: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("VECTOR_DB_PATH", ".molprop/vectors")) if os.getenv("VECTOR_DB_PATH") else None
    )

    def __post_init__(self) -> None:
        """Ensure directory paths exist."""
        self.library_db_path = self.library_db_path.expanduser()
        self.library_db_path.parent.mkdir(parents=True, exist_ok=True)

        if self.vector_db_path:
            self.vector_db_path = self.vector_db_path.expanduser()
            self.vector_db_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main application configuration (composition of all configs)."""

    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Paths
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data")

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        self.root_dir = self.root_dir.expanduser()
        self.data_dir = self.data_dir.expanduser()

    @classmethod
    def from_env(cls) -> Config:
        """Create configuration from environment variables."""
        return cls()


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global application configuration."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
