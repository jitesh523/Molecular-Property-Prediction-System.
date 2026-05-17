"""
Substructure search via SMARTS / SMILES query patterns.

Supports both arbitrary lists of candidate SMILES and a "search the library"
shortcut that loads compounds from the persistent SQLite store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem


@dataclass
class SubstructureMatch:
    smiles: str
    name: Optional[str]
    n_matches: int
    atom_indices: list[list[int]]


def _compile_query(query: str) -> Optional[Chem.Mol]:
    """Try SMARTS first, then SMILES."""
    q = Chem.MolFromSmarts(query)
    if q is None:
        q = Chem.MolFromSmiles(query)
    return q


def substructure_search(
    query: str,
    candidates: list[dict],
    limit: int = 100,
) -> tuple[list[SubstructureMatch], dict]:
    """
    Return all candidates whose molecules contain the query substructure.

    `candidates` is a list of `{"smiles": ..., "name": ...}` dicts (name optional).
    Returns the matches plus a stats dict.
    """
    q = _compile_query(query)
    if q is None:
        return [], {"error": f"Invalid query pattern: '{query}'"}

    matches: list[SubstructureMatch] = []
    n_searched = 0
    n_invalid = 0
    for c in candidates:
        smi = c.get("smiles", "")
        if not smi:
            continue
        n_searched += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_invalid += 1
            continue
        m = mol.GetSubstructMatches(q, uniquify=True)
        if m:
            matches.append(
                SubstructureMatch(
                    smiles=smi,
                    name=c.get("name"),
                    n_matches=len(m),
                    atom_indices=[list(idx) for idx in m],
                )
            )
            if len(matches) >= limit:
                break

    return matches, {
        "n_candidates": len(candidates),
        "n_searched": n_searched,
        "n_invalid": n_invalid,
        "n_matches": len(matches),
    }
