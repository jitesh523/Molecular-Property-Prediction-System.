"""Tests for security utilities."""

import pytest

from molprop.utils.security import (
    constant_time_compare,
    generate_api_key,
    hash_password,
    is_safe_sql_identifier,
    rate_limit_key,
    safe_filename,
    sanitize_log_data,
    sanitize_string,
    validate_api_key,
    verify_password,
)


class TestSanitizeString:
    """Test string sanitization."""

    def test_sanitize_normal_string(self):
        """Normal strings should pass through unchanged."""
        result = sanitize_string("normal string")
        assert result == "normal string"

    def test_sanitize_removes_null_bytes(self):
        """Null bytes should be removed."""
        result = sanitize_string("test\x00string")
        assert "\x00" not in result
        assert "teststring" == result

    def test_sanitize_removes_control_characters(self):
        """Control characters should be removed."""
        result = sanitize_string("test\x01\x02string")
        assert "teststring" == result

    def test_sanitize_preserves_newlines_tabs(self):
        """Newlines and tabs should be preserved."""
        result = sanitize_string("test\nline\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_strips_whitespace(self):
        """Leading/trailing whitespace should be removed."""
        result = sanitize_string("  test string  ")
        assert result == "test string"

    def test_sanitize_enforces_max_length(self):
        """Should raise error if exceeds max length."""
        with pytest.raises(ValueError):
            sanitize_string("x" * 1001, max_length=1000)


class TestSafeFilename:
    """Test safe filename generation."""

    def test_safe_filename_removes_path_separators(self):
        """Path separators should be removed."""
        result = safe_filename("path/to/file.txt")
        assert "/" not in result
        assert "path_to_file.txt" == result

    def test_safe_filename_removes_special_characters(self):
        """Special characters should be removed."""
        result = safe_filename("file@#$%.txt")
        assert result == "file.txt"

    def test_safe_filename_preserves_extension(self):
        """File extension should be preserved."""
        result = safe_filename("my_file.pdf")
        assert result.endswith(".pdf")

    def test_safe_filename_limits_length(self):
        """Filename should be limited to 255 characters."""
        long_name = "a" * 300 + ".txt"
        result = safe_filename(long_name)
        assert len(result) <= 255

    def test_safe_filename_handles_empty(self):
        """Empty/special-char-only names should get default."""
        result = safe_filename("@#$%")
        assert result == "file"


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password_generates_salt(self):
        """Should generate a salt if not provided."""
        hashed, salt = hash_password("password123")
        assert salt is not None
        assert len(salt) > 0
        assert hashed != "password123"

    def test_hash_password_uses_provided_salt(self):
        """Should use provided salt."""
        salt = "test_salt_123"
        hashed, returned_salt = hash_password("password123", salt)
        assert returned_salt == salt

    def test_verify_password_with_correct_password(self):
        """Should verify correct password."""
        password = "mypassword123"
        hashed, salt = hash_password(password)
        assert verify_password(password, hashed, salt) is True

    def test_verify_password_with_incorrect_password(self):
        """Should reject incorrect password."""
        password = "mypassword123"
        hashed, salt = hash_password(password)
        assert verify_password("wrongpassword", hashed, salt) is False

    def test_different_salts_produce_different_hashes(self):
        """Different salts should produce different hashes."""
        password = "password123"
        hash1, salt1 = hash_password(password, "salt1")
        hash2, salt2 = hash_password(password, "salt2")
        assert hash1 != hash2


class TestAPIKeyGeneration:
    """Test API key generation and validation."""

    def test_generate_api_key_default(self):
        """Should generate key with default prefix."""
        key = generate_api_key()
        assert key.startswith("sk_")

    def test_generate_api_key_custom_prefix(self):
        """Should generate key with custom prefix."""
        key = generate_api_key(prefix="test")
        assert key.startswith("test_")

    def test_generate_api_key_length(self):
        """Generated key should have specified length."""
        key = generate_api_key(length=16)
        parts = key.split("_")
        assert len(parts) == 2

    def test_validate_api_key_valid(self):
        """Should validate correct API key format."""
        key = generate_api_key()
        assert validate_api_key(key) is True

    def test_validate_api_key_invalid(self):
        """Should reject invalid API key format."""
        assert validate_api_key("invalid_key_format!") is False
        assert validate_api_key("no_underscore") is False


class TestConstantTimeCompare:
    """Test constant-time comparison."""

    def test_constant_time_compare_equal(self):
        """Should return True for equal strings."""
        assert constant_time_compare("test", "test") is True

    def test_constant_time_compare_not_equal(self):
        """Should return False for different strings."""
        assert constant_time_compare("test", "other") is False

    def test_constant_time_compare_empty_strings(self):
        """Should handle empty strings."""
        assert constant_time_compare("", "") is True
        assert constant_time_compare("", "test") is False


class TestSQLIdentifierValidation:
    """Test SQL identifier validation."""

    def test_is_safe_sql_identifier_valid(self):
        """Valid identifiers should be accepted."""
        assert is_safe_sql_identifier("table_name") is True
        assert is_safe_sql_identifier("Column_1") is True
        assert is_safe_sql_identifier("test-table") is True

    def test_is_safe_sql_identifier_invalid(self):
        """Invalid identifiers should be rejected."""
        assert is_safe_sql_identifier("table; DROP TABLE users") is False
        assert is_safe_sql_identifier("table.name") is False
        assert is_safe_sql_identifier("table'name") is False
        assert is_safe_sql_identifier("table name") is False


class TestRateLimitKey:
    """Test rate limit key generation."""

    def test_rate_limit_key_format(self):
        """Should generate correctly formatted key."""
        key = rate_limit_key("client123", "predict")
        assert key.startswith("ratelimit:")
        assert "client123" in key
        assert "predict" in key

    def test_rate_limit_key_same_window(self):
        """Same client/op in same window should return same key."""
        key1 = rate_limit_key("client123", "predict", window=60)
        key2 = rate_limit_key("client123", "predict", window=60)
        assert key1 == key2

    def test_rate_limit_key_different_clients(self):
        """Different clients should get different keys."""
        key1 = rate_limit_key("client1", "predict")
        key2 = rate_limit_key("client2", "predict")
        assert key1 != key2


class TestSanitizeLogData:
    """Test sensitive data sanitization for logs."""

    def test_sanitize_log_data_password(self):
        """Password fields should be redacted."""
        data = {"username": "user", "password": "secret123"}
        result = sanitize_log_data(data)
        assert result["username"] == "user"
        assert result["password"] == "***REDACTED***"

    def test_sanitize_log_data_token(self):
        """Token fields should be redacted."""
        data = {"auth_token": "abc123", "username": "user"}
        result = sanitize_log_data(data)
        assert result["auth_token"] == "***REDACTED***"

    def test_sanitize_log_data_nested(self):
        """Should recursively sanitize nested dicts."""
        data = {
            "user": {"password": "secret", "name": "John"},
            "api_key": "key123",
        }
        result = sanitize_log_data(data)
        assert result["user"]["password"] == "***REDACTED***"
        assert result["user"]["name"] == "John"
        assert result["api_key"] == "***REDACTED***"

    def test_sanitize_log_data_case_insensitive(self):
        """Should detect sensitive keys case-insensitively."""
        data = {"PASSWORD": "secret", "ApiKey": "key123"}
        result = sanitize_log_data(data)
        assert result["PASSWORD"] == "***REDACTED***"
        assert result["ApiKey"] == "***REDACTED***"
