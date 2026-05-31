"""
Response formatting and standardization utilities.

Provides consistent response structures, error formatting, and
JSON serialization for API responses.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from molprop.exceptions import MolpropError


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder supporting common types."""

    def default(self, obj: Any) -> Any:
        """Encode objects to JSON-serializable format."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, Enum):
            return obj.value
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


def success_response(
    data: Any,
    message: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create a standardized success response.

    Args:
        data: Response data.
        message: Optional success message.
        meta: Optional metadata (count, timestamp, etc).

    Returns:
        Formatted response dictionary.
    """
    response: dict[str, Any] = {
        "status": "success",
        "data": data,
    }

    if message:
        response["message"] = message

    if meta:
        response["meta"] = meta

    response["timestamp"] = datetime.utcnow().isoformat()

    return response


def error_response(
    error: MolpropError | Exception,
    status_code: int = 500,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        error: Exception that occurred.
        status_code: HTTP status code.
        request_id: Optional request ID for tracing.

    Returns:
        Formatted error response.
    """
    if isinstance(error, MolpropError):
        error_code = error.code
        message = error.message
        details = {}
        if hasattr(error, "details"):
            details = error.details  # type: ignore
    else:
        error_code = "INTERNAL_ERROR"
        message = str(error)
        details = {}

    response: dict[str, Any] = {
        "status": "error",
        "error": {
            "code": error_code,
            "message": message,
        },
    }

    if details:
        response["error"]["details"] = details

    if request_id:
        response["request_id"] = request_id

    response["timestamp"] = datetime.utcnow().isoformat()

    return response


def paginated_response(
    items: list[Any],
    total: int,
    page: int,
    page_size: int,
    message: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a paginated response.

    Args:
        items: List of items for this page.
        total: Total number of items.
        page: Current page number (1-indexed).
        page_size: Items per page.
        message: Optional message.

    Returns:
        Paginated response with metadata.
    """
    total_pages = (total + page_size - 1) // page_size

    return success_response(
        data=items,
        message=message,
        meta={
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            }
        },
    )


def batch_response(
    results: list[Any],
    errors: Optional[list[tuple[int, str]]] = None,
    message: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a batch operation response.

    Args:
        results: List of successful results.
        errors: Optional list of (index, error_message) tuples.
        message: Optional message.

    Returns:
        Batch response with results and errors.
    """
    response = success_response(
        data={
            "successful": len(results),
            "failed": len(errors) if errors else 0,
            "results": results,
        },
        message=message,
    )

    if errors:
        response["data"]["errors"] = [
            {"index": idx, "message": msg} for idx, msg in errors
        ]

    return response


def to_json(obj: Any, pretty: bool = False) -> str:
    """
    Serialize object to JSON using custom encoder.

    Args:
        obj: Object to serialize.
        pretty: Whether to pretty-print.

    Returns:
        JSON string.
    """
    indent = 2 if pretty else None
    return json.dumps(obj, cls=JSONEncoder, indent=indent)


def from_json(data: str) -> Any:
    """
    Deserialize JSON string.

    Args:
        data: JSON string.

    Returns:
        Deserialized object.
    """
    return json.loads(data)
