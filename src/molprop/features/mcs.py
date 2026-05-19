"""
Maximum Common Substructure (MCS) computation between two molecules.

Wraps RDKit's `rdFMCS.FindMCS` with sensible defaults for medicinal-chemistry
comparisons (ring matches must be exact; atoms are compared by element).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdFMCS


@dataclass
class MCSResult:
    smarts: Optional[str]
    n_atoms: int
    n_bonds: int
    canceled: bool
    a_atom_indices: list[int]
    b_atom_indices: list[int]
    a_coverage: float  # fraction of A's heavy atoms covered
    b_coverage: float  # fraction of B's heavy atoms covered

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def find_mcs(
    smiles_a: str,
    smiles_b: str,
    timeout: int = 5,
    complete_rings_only: bool = True,
    match_valences: bool = False,
    ring_matches_ring_only: bool = True,
) -> Optional[MCSResult]:
    """
    Find the maximum common substructure between two molecules.

    Returns ``None`` if either SMILES is invalid.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None

    res = rdFMCS.FindMCS(
        [mol_a, mol_b],
        timeout=timeout,
        completeRingsOnly=complete_rings_only,
        ringMatchesRingOnly=ring_matches_ring_only,
        matchValences=match_valences,
    )

    smarts = res.smartsString if res.numAtoms > 0 else None
    a_match: tuple[int, ...] = ()
    b_match: tuple[int, ...] = ()
    if smarts:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            a_match = mol_a.GetSubstructMatch(patt) or ()
            b_match = mol_b.GetSubstructMatch(patt) or ()

    a_heavy = mol_a.GetNumHeavyAtoms() or 1
    b_heavy = mol_b.GetNumHeavyAtoms() or 1

    return MCSResult(
        smarts=smarts,
        n_atoms=res.numAtoms,
        n_bonds=res.numBonds,
        canceled=bool(res.canceled),
        a_atom_indices=list(a_match),
        b_atom_indices=list(b_match),
        a_coverage=round(res.numAtoms / a_heavy, 4),
        b_coverage=round(res.numAtoms / b_heavy, 4),
    )
