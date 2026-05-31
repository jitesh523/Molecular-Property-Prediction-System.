"""
Security utilities for molprop.

Provides functions for input sanitization, SQL injection prevention,
secure hashing, and other security-related operations.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
from typing import Optional

log = logging.getLogger(__name__)


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input string.

    Args:
        text: String to sanitize.
        max_length: Maximum allowed length.

    Returns:
        Sanitized string.

    Raises:
        ValueError: If string exceeds max length.
    """
    if len(text) > max_length:
        raise ValueError(f"String exceeds maximum length of {max_length}")

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove control characters except newline/tab
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")

    return text.strip()


def safe_filename(filename: str) -> str:
    """
    Convert filename to safe format.

    Args:
        filename: Original filename.

    Returns:
        Safe filename with no path traversal or special chars.
    """
    # Remove path separators
    filename = filename.replace("/", "_").replace("\\", "_")

    # Remove special characters, keep only alphanumeric, dots, hyphens, underscores
    filename = re.sub(r"[^a-zA-Z0-9._-]", "", filename)

    # Prevent empty filename
    if not filename:
        filename = "file"

    # Limit length
    if len(filename) > 255:
        # Keep extension
        if "." in filename:
            name, ext = filename.rsplit(".", 1)
            filename = name[: 255 - len(ext) - 1] + "." + ext
        else:
            filename = filename[:255]

    return filename


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password using PBKDF2.

    Args:
        password: Password to hash.
        salt: Optional salt. If None, generates a new one.

    Returns:
        Tuple of (hashed_password, salt).
    """
    if salt is None:
        salt = secrets.token_hex(16)

    hash_obj = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    hashed = hash_obj.hex()

    return hashed, salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """
    Verify a password against hash.

    Args:
        password: Password to verify.
        hashed: Stored hash.
        salt: Salt used for hashing.

    Returns:
        True if password matches.
    """
    new_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(new_hash, hashed)


def generate_api_key(prefix: str = "sk", length: int = 32) -> str:
    """
    Generate a secure API key.

    Args:
        prefix: Key prefix.
        length: Length of random part.

    Returns:
        Generated API key.
    """
    random_part = secrets.token_urlsafe(length)
    return f"{prefix}_{random_part}"


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.

    Args:
        api_key: API key to validate.

    Returns:
        True if valid format.
    """
    # Format: prefix_base64urlsafe_characters
    return bool(re.match(r"^[a-zA-Z0-9_-]+_[a-zA-Z0-9_-]+$", api_key))


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.

    Args:
        a: First string.
        b: Second string.

    Returns:
        True if strings are equal.
    """
    return hmac.compare_digest(a, b)


def is_safe_sql_identifier(identifier: str) -> bool:
    """
    Check if string is a safe SQL identifier (no injection risk).

    Args:
        identifier: Identifier to check.

    Returns:
        True if safe.
    """
    # Only alphanumeric, underscores, and hyphens allowed
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", identifier))


def rate_limit_key(client_id: str, operation: str, window: int = 60) -> str:
    """
    Generate a rate limit key for caching.

    Args:
        client_id: Client identifier.
        operation: Operation name.
        window: Time window in seconds.

    Returns:
        Rate limit key.
    """
    import time

    window_num = int(time.time() / window)
    return f"ratelimit:{client_id}:{operation}:{window_num}"


def sanitize_log_data(data: dict) -> dict:
    """
    Remove sensitive data from logs.

    Args:
        data: Data dictionary to sanitize.

    Returns:
        Sanitized dictionary.
    """
    sensitive_keys = {
        "password",
        "token",
        "api_key",
        "secret",
        "authorization",
        "auth",
        "cookie",
        "credit_card",
        "ssn",
    }

    sanitized = {}
    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        else:
            sanitized[key] = value

    return sanitized
