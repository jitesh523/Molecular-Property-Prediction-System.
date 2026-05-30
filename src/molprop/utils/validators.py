"""
Input validation utilities for molprop.

Provides validators for common inputs like SMILES strings,
molecular properties, and API parameters.
"""

from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem

from molprop.exceptions import InvalidSMILESError, ValidationError


def validate_smiles(smiles: str, allow_empty: bool = False) -> str:
    """
    Validate and canonicalize a SMILES string.

    Args:
        smiles: The SMILES string to validate.
        allow_empty: Whether to allow empty strings.

    Returns:
        The canonicalized SMILES string.

    Raises:
        InvalidSMILESError: If the SMILES is invalid.
        ValidationError: If the string is empty and not allowed.
    """
    if not smiles:
        if allow_empty:
            return ""
        raise ValidationError("SMILES string cannot be empty", field="smiles")

    smiles = smiles.strip()
    if not smiles:
        raise ValidationError("SMILES string cannot be empty or whitespace only", field="smiles")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSMILESError(smiles, "RDKit failed to parse SMILES")

    # Return canonical SMILES
    try:
        canonical = Chem.MolToSmiles(mol)
        return canonical
    except Exception as e:
        raise InvalidSMILESError(smiles, f"Failed to canonicalize: {str(e)}")


def validate_smiles_list(smiles_list: list[str], min_length: int = 1, max_length: int = 1000) -> list[str]:
    """
    Validate a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.
        min_length: Minimum number of SMILES required.
        max_length: Maximum number of SMILES allowed.

    Returns:
        List of canonicalized SMILES strings.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(smiles_list, list):
        raise ValidationError("Input must be a list of SMILES strings", field="smiles_list")

    if len(smiles_list) < min_length:
        raise ValidationError(
            f"List must have at least {min_length} SMILES, got {len(smiles_list)}", field="smiles_list"
        )

    if len(smiles_list) > max_length:
        raise ValidationError(
            f"List cannot exceed {max_length} SMILES, got {len(smiles_list)}", field="smiles_list"
        )

    validated = []
    for i, smiles in enumerate(smiles_list):
        try:
            validated.append(validate_smiles(smiles))
        except InvalidSMILESError as e:
            raise ValidationError(f"Invalid SMILES at index {i}: {e.message}", field=f"smiles_list[{i}]")

    return validated


def validate_project_name(project: str) -> str:
    """
    Validate a project name (alphanumeric, underscores, hyphens).

    Args:
        project: The project name to validate.

    Returns:
        The validated project name.

    Raises:
        ValidationError: If the project name is invalid.
    """
    if not project:
        raise ValidationError("Project name cannot be empty", field="project")

    if not re.match(r"^[a-zA-Z0-9_-]+$", project):
        raise ValidationError(
            "Project name can only contain alphanumeric characters, underscores, and hyphens", field="project"
        )

    if len(project) > 255:
        raise ValidationError("Project name cannot exceed 255 characters", field="project")

    return project


def validate_tags(tags: Optional[list[str]]) -> list[str]:
    """
    Validate a list of tags.

    Args:
        tags: List of tags to validate.

    Returns:
        The validated tags list (deduplicated and cleaned).

    Raises:
        ValidationError: If tags are invalid.
    """
    if tags is None:
        return []

    if not isinstance(tags, list):
        raise ValidationError("Tags must be a list of strings", field="tags")

    if len(tags) > 100:
        raise ValidationError("Cannot have more than 100 tags", field="tags")

    cleaned = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValidationError(f"All tags must be strings, got {type(tag).__name__}", field="tags")

        tag = tag.strip()
        if not tag:
            continue

        if len(tag) > 100:
            raise ValidationError(f"Tag cannot exceed 100 characters: {tag}", field="tags")

        cleaned.append(tag)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for tag in cleaned:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)

    return unique


def validate_batch_size(batch_size: int, min_size: int = 1, max_size: int = 1000) -> int:
    """
    Validate batch size.

    Args:
        batch_size: The batch size to validate.
        min_size: Minimum allowed batch size.
        max_size: Maximum allowed batch size.

    Returns:
        The validated batch size.

    Raises:
        ValidationError: If batch size is invalid.
    """
    if not isinstance(batch_size, int):
        raise ValidationError(f"Batch size must be an integer, got {type(batch_size).__name__}", field="batch_size")

    if batch_size < min_size or batch_size > max_size:
        raise ValidationError(
            f"Batch size must be between {min_size} and {max_size}, got {batch_size}", field="batch_size"
        )

    return batch_size
