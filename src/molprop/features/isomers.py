"""
Tautomer and stereoisomer enumeration.

Wraps RDKit's `MolStandardize.rdMolStandardize.TautomerEnumerator` and
`EnumerateStereoisomers` into a convenience layer that returns canonical
SMILES alongside a tautomer "score" (RDKit's energy-like ranking) and the
canonical tautomer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass
class IsomerResult:
    smiles: str
    canonical_smiles: Optional[str]
    is_canonical: bool


@dataclass
class IsomerEnumeration:
    input_smiles: str
    canonical_input_smiles: Optional[str]
    canonical_tautomer: Optional[str]
    tautomers: list[IsomerResult]
    stereoisomers: list[IsomerResult]
    n_tautomers: int
    n_stereoisomers: int

    def to_dict(self) -> dict:
        return {
            "input_smiles": self.input_smiles,
            "canonical_input_smiles": self.canonical_input_smiles,
            "canonical_tautomer": self.canonical_tautomer,
            "tautomers": [t.__dict__ for t in self.tautomers],
            "stereoisomers": [s.__dict__ for s in self.stereoisomers],
            "n_tautomers": self.n_tautomers,
            "n_stereoisomers": self.n_stereoisomers,
        }


def enumerate_isomers(
    smiles: str,
    max_tautomers: int = 25,
    max_stereoisomers: int = 16,
) -> Optional[IsomerEnumeration]:
    """
    Return tautomers + stereoisomers for a SMILES.

    Returns ``None`` for invalid input. Limits are enforced to avoid
    combinatorial explosion on heavily stereogenic molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical_input = Chem.MolToSmiles(mol)

    # ── Tautomers ────────────────────────────────────────────────────────────
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    enumerator.SetMaxTransforms(1000)

    taut_results: list[IsomerResult] = []
    canonical_taut: Optional[str] = None
    try:
        canonical_taut_mol = enumerator.Canonicalize(mol)
        canonical_taut = Chem.MolToSmiles(canonical_taut_mol)
    except Exception:
        canonical_taut = None

    try:
        taut_mols = list(enumerator.Enumerate(mol))
        for t in taut_mols[:max_tautomers]:
            smi = Chem.MolToSmiles(t)
            taut_results.append(
                IsomerResult(
                    smiles=smi,
                    canonical_smiles=canonical_taut,
                    is_canonical=(smi == canonical_taut),
                )
            )
    except Exception:
        # Fall back to the input itself
        taut_results = [
            IsomerResult(
                smiles=canonical_input,
                canonical_smiles=canonical_taut,
                is_canonical=True,
            )
        ]

    # ── Stereoisomers ────────────────────────────────────────────────────────
    stereo_results: list[IsomerResult] = []
    try:
        opts = StereoEnumerationOptions(
            tryEmbedding=False,
            unique=True,
            onlyUnassigned=True,
            maxIsomers=max_stereoisomers,
        )
        for s in EnumerateStereoisomers(mol, options=opts):
            smi = Chem.MolToSmiles(s)
            stereo_results.append(
                IsomerResult(
                    smiles=smi,
                    canonical_smiles=canonical_input,
                    is_canonical=(smi == canonical_input),
                )
            )
    except Exception:
        stereo_results = [
            IsomerResult(
                smiles=canonical_input,
                canonical_smiles=canonical_input,
                is_canonical=True,
            )
        ]

    return IsomerEnumeration(
        input_smiles=smiles,
        canonical_input_smiles=canonical_input,
        canonical_tautomer=canonical_taut,
        tautomers=taut_results,
        stereoisomers=stereo_results,
        n_tautomers=len(taut_results),
        n_stereoisomers=len(stereo_results),
    )
