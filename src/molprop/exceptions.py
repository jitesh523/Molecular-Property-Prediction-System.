"""
Custom exception types for molprop.

Provides typed, specific exceptions for different error categories
to enable better error handling and debugging.
"""

from __future__ import annotations

from typing import Any, Optional


class MolpropError(Exception):
    """Base exception for all molprop errors."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class ConfigurationError(MolpropError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIGURATION_ERROR")


class ValidationError(MolpropError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        self.field = field
        super().__init__(message, code="VALIDATION_ERROR")


class CheminformaticsError(MolpropError):
    """Raised when cheminformatics operations fail."""

    def __init__(self, message: str, smiles: Optional[str] = None) -> None:
        self.smiles = smiles
        super().__init__(message, code="CHEMINFORMATICS_ERROR")


class InvalidSMILESError(CheminformaticsError):
    """Raised when SMILES parsing fails."""

    def __init__(self, smiles: str, reason: Optional[str] = None) -> None:
        msg = f"Invalid SMILES: '{smiles}'"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg, smiles=smiles)
        self.code = "INVALID_SMILES"


class ModelError(MolpropError):
    """Raised when model operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="MODEL_ERROR")


class ModelLoadError(ModelError):
    """Raised when model loading fails."""

    def __init__(self, model_type: str, path: str, reason: Optional[str] = None) -> None:
        msg = f"Failed to load {model_type} model from {path}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
        self.code = "MODEL_LOAD_ERROR"


class InferenceError(ModelError):
    """Raised when inference fails."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.details = details or {}
        self.code = "INFERENCE_ERROR"


class RateLimitError(MolpropError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None) -> None:
        self.retry_after = retry_after
        super().__init__(message, code="RATE_LIMIT_EXCEEDED")


class APIError(MolpropError):
    """Raised when API operations fail."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "API_ERROR",
        details: Optional[dict] = None,
    ) -> None:
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message, code=code)


class NotFoundError(APIError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource: str, identifier: Optional[str] = None) -> None:
        msg = f"{resource} not found"
        if identifier:
            msg += f": {identifier}"
        super().__init__(msg, status_code=404, code="NOT_FOUND")


class BadRequestError(APIError):
    """Raised when the request is malformed or invalid."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, status_code=400, code="BAD_REQUEST", details=details)


class ServerError(APIError):
    """Raised when an internal server error occurs."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message, status_code=500, code="INTERNAL_SERVER_ERROR", details=details)


class TimeoutError(MolpropError):
    """Raised when an operation exceeds the timeout."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(message, code="TIMEOUT")


class StorageError(MolpropError):
    """Raised when storage/persistence operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None) -> None:
        self.operation = operation
        super().__init__(message, code="STORAGE_ERROR")


# Error mapping for HTTP status codes
HTTP_STATUS_TO_EXCEPTION: dict[int, type[APIError]] = {
    400: BadRequestError,
    404: NotFoundError,
    500: ServerError,
}


def get_exception_for_status(status_code: int, message: str, details: Optional[dict] = None) -> APIError:
    """Get the appropriate exception for an HTTP status code."""
    exc_class = HTTP_STATUS_TO_EXCEPTION.get(status_code, APIError)
    if issubclass(exc_class, APIError):
        return exc_class(message, status_code=status_code, details=details)  # type: ignore
    return exc_class(message, details=details)  # type: ignore
