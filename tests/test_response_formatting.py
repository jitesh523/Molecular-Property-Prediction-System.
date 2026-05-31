"""Tests for response formatting utilities."""

import pytest
from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID

from molprop.utils.response import (
    JSONEncoder,
    batch_response,
    error_response,
    from_json,
    paginated_response,
    success_response,
    to_json,
)
from molprop.exceptions import ValidationError


class TestJSONEncoder:
    """Test custom JSON encoder."""

    def test_encode_datetime(self):
        """Should encode datetime as ISO format."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        encoded = to_json({"timestamp": dt})
        assert "2024-01-15T10:30:45" in encoded

    def test_encode_uuid(self):
        """Should encode UUID as string."""
        uuid_obj = UUID("12345678-1234-5678-1234-567812345678")
        encoded = to_json({"id": uuid_obj})
        assert "12345678-1234-5678-1234-567812345678" in encoded

    def test_encode_decimal(self):
        """Should encode Decimal as float."""
        dec = Decimal("10.55")
        encoded = to_json({"value": dec})
        assert "10.55" in encoded

    def test_encode_enum(self):
        """Should encode Enum as value."""

        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        encoded = to_json({"status": Status.ACTIVE})
        assert "active" in encoded


class TestSuccessResponse:
    """Test success response formatting."""

    def test_success_response_basic(self):
        """Should create basic success response."""
        response = success_response({"result": "ok"})
        assert response["status"] == "success"
        assert response["data"] == {"result": "ok"}
        assert "timestamp" in response

    def test_success_response_with_message(self):
        """Should include message if provided."""
        response = success_response({"result": "ok"}, message="Operation successful")
        assert response["message"] == "Operation successful"

    def test_success_response_with_metadata(self):
        """Should include metadata if provided."""
        meta = {"count": 10, "version": "1.0"}
        response = success_response({"data": []}, meta=meta)
        assert response["meta"] == meta


class TestErrorResponse:
    """Test error response formatting."""

    def test_error_response_with_molprop_error(self):
        """Should format MolpropError correctly."""
        error = ValidationError("Invalid input", field="name")
        response = error_response(error, status_code=400)
        assert response["status"] == "error"
        assert response["error"]["code"] == "VALIDATION_ERROR"
        assert "Invalid input" in response["error"]["message"]

    def test_error_response_with_generic_error(self):
        """Should format generic exceptions."""
        error = ValueError("Test error")
        response = error_response(error, status_code=500)
        assert response["status"] == "error"
        assert response["error"]["code"] == "INTERNAL_ERROR"
        assert "Test error" in response["error"]["message"]

    def test_error_response_with_request_id(self):
        """Should include request ID if provided."""
        error = ValueError("Test")
        response = error_response(error, request_id="req_123")
        assert response["request_id"] == "req_123"


class TestPaginatedResponse:
    """Test paginated response formatting."""

    def test_paginated_response_first_page(self):
        """Should format first page correctly."""
        items = [{"id": 1}, {"id": 2}]
        response = paginated_response(items, total=100, page=1, page_size=2)

        assert response["status"] == "success"
        assert response["data"] == items
        assert response["meta"]["pagination"]["page"] == 1
        assert response["meta"]["pagination"]["total"] == 100
        assert response["meta"]["pagination"]["total_pages"] == 50
        assert response["meta"]["pagination"]["has_next"] is True
        assert response["meta"]["pagination"]["has_prev"] is False

    def test_paginated_response_last_page(self):
        """Should handle last page correctly."""
        response = paginated_response([], total=50, page=25, page_size=2)
        assert response["meta"]["pagination"]["has_next"] is False
        assert response["meta"]["pagination"]["has_prev"] is True

    def test_paginated_response_single_page(self):
        """Should handle single-page results."""
        response = paginated_response([], total=10, page=1, page_size=20)
        assert response["meta"]["pagination"]["total_pages"] == 1
        assert response["meta"]["pagination"]["has_next"] is False


class TestBatchResponse:
    """Test batch operation response formatting."""

    def test_batch_response_success(self):
        """Should format successful batch response."""
        results = [{"id": 1, "status": "ok"}, {"id": 2, "status": "ok"}]
        response = batch_response(results)

        assert response["status"] == "success"
        assert response["data"]["successful"] == 2
        assert response["data"]["failed"] == 0

    def test_batch_response_with_errors(self):
        """Should include error information."""
        results = [{"id": 1, "status": "ok"}]
        errors = [(1, "Invalid input"), (2, "Not found")]
        response = batch_response(results, errors=errors)

        assert response["data"]["successful"] == 1
        assert response["data"]["failed"] == 2
        assert len(response["data"]["errors"]) == 2
        assert response["data"]["errors"][0]["index"] == 1

    def test_batch_response_with_message(self):
        """Should include message if provided."""
        response = batch_response([], message="Batch processing complete")
        assert response["message"] == "Batch processing complete"


class TestJsonSerialization:
    """Test JSON serialization/deserialization."""

    def test_to_json_default(self):
        """Should serialize to compact JSON by default."""
        data = {"name": "test", "value": 123}
        json_str = to_json(data)
        assert "\n" not in json_str  # No newlines in compact format
        assert '"name"' in json_str

    def test_to_json_pretty(self):
        """Should serialize to pretty-printed JSON."""
        data = {"name": "test", "value": 123}
        json_str = to_json(data, pretty=True)
        assert "\n" in json_str  # Should have newlines

    def test_from_json(self):
        """Should deserialize JSON correctly."""
        json_str = '{"name": "test", "value": 123}'
        data = from_json(json_str)
        assert data["name"] == "test"
        assert data["value"] == 123

    def test_json_roundtrip(self):
        """Data should survive JSON roundtrip."""
        original = {"items": [1, 2, 3], "name": "test", "active": True}
        json_str = to_json(original)
        restored = from_json(json_str)
        assert restored == original
