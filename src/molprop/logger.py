"""
Logging configuration for molprop.

Sets up structured logging with different levels for development,
testing, and production environments.
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional

# JSON logging for production, human-readable for dev
try:
    import json_logging

    HAS_JSON_LOGGING = True
except ImportError:
    HAS_JSON_LOGGING = False


def setup_logging(
    level: Optional[str] = None,
    use_json: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to env var or INFO.
        use_json: Whether to use JSON formatting (useful for production).
        log_file: Optional path to write logs to file.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    # Remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure formatters
    if use_json and HAS_JSON_LOGGING:
        json_logging.init_non_web(custom_json_handler=None)
        root_logger.setLevel(level)
    else:
        # Human-readable format
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

        root_logger.setLevel(level)

    # File handler if requested
    if log_file:
        log_file = Path(log_file).expanduser()
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        if use_json and HAS_JSON_LOGGING:
            # JSON format for file
            file_formatter = logging.Formatter("%(message)s")
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Suppress verbose third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("rdkit").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Initialize logging on module import
_log_level = os.getenv("LOG_LEVEL", "INFO")
_use_json = os.getenv("LOG_FORMAT", "text").lower() == "json"
_log_file = os.getenv("LOG_FILE")

setup_logging(level=_log_level, use_json=_use_json, log_file=_log_file)
