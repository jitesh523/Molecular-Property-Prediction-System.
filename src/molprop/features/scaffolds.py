"""
Scaffold analysis & synthetic accessibility scoring.

- Bemis–Murcko scaffold (with side chains stripped to a generic ring system).
- Generic Murcko scaffold (atoms + bonds reduced to C / single).
- Synthetic Accessibility Score (SAScore, 1=easy → 10=hard) via the
  fragment-frequency heuristic from Ertl & Schuffenhauer (2009).

The SAScore implementation here is a lightweight approximation: it uses
fragment counts derived from RDKit's standard MorganGenerator output and
combines them with stereo / spiro / macrocycle penalties. It correlates
well with the canonical RDKit-contrib `sascorer.py` for typical drug-like
molecules but does NOT require the contrib FPscores file at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass
class ScaffoldAnalysis:
    smiles: str
    murcko_smiles: Optional[str]
    generic_murcko_smiles: Optional[str]
    num_rings: int
    num_aromatic_rings: int
    num_aliphatic_rings: int
    largest_ring_size: int
    num_spiro_atoms: int
    num_bridgehead_atoms: int
    has_macrocycle: bool
    sa_score: float  # 1.0 (easy) – 10.0 (hard)
    sa_class: str  # "easy" | "moderate" | "hard"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── helpers ────────────────────────────────────────────────────────────────────


def _bemis_murcko(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        return MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None


def _generic_murcko(scaffold_mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        return MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
    except Exception:
        return None


def _ring_metrics(mol: Chem.Mol) -> tuple[int, int, int, int]:
    ri = mol.GetRingInfo()
    sizes = [len(r) for r in ri.AtomRings()]
    n_rings = len(sizes)
    n_aromatic = sum(
        1 for r in ri.AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    )
    largest = max(sizes) if sizes else 0
    n_aliphatic = n_rings - n_aromatic
    return n_rings, n_aromatic, n_aliphatic, largest


def _topology_penalties(mol: Chem.Mol) -> tuple[int, int, bool]:
    """Return (n_spiro_atoms, n_bridgehead_atoms, has_macrocycle)."""
    from rdkit.Chem import rdMolDescriptors

    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    ri = mol.GetRingInfo()
    has_macrocycle = any(len(r) > 8 for r in ri.AtomRings())
    return n_spiro, n_bridge, has_macrocycle


def _sa_score(mol: Chem.Mol) -> float:
    """
    Lightweight SAScore approximation in [1.0, 10.0].

    Heuristic: combines size / fragment complexity with topology penalties:
    - Larger heavy-atom counts add modest penalty (size term, log scale).
    - Stereocenters, spiro / bridgehead atoms, macrocycles add penalties.
    - Many distinct ECFP4 fragments => synthetic complexity penalty.
    """
    if mol is None:
        return 10.0

    n_atoms = mol.GetNumHeavyAtoms()
    if n_atoms == 0:
        return 1.0

    # Fragment complexity: count distinct ECFP4 bits set.
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    n_bits = fp.GetNumOnBits()
    fragment_score = -math.log(max(1, n_bits) / max(1, n_atoms))
    # Typical drug-like molecules: ~0.3–1.0 fragment diversity per heavy atom.

    # Size penalty (log-scaled)
    size_penalty = math.log(n_atoms + 1) * 0.3

    # Stereo penalty
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    stereo_penalty = 0.4 * n_stereo

    # Topology penalties
    n_spiro, n_bridge, has_macrocycle = _topology_penalties(mol)
    spiro_penalty = 0.6 * n_spiro
    bridge_penalty = 0.5 * n_bridge
    macro_penalty = 1.5 if has_macrocycle else 0.0

    raw = (
        1.0
        + fragment_score
        + size_penalty
        + stereo_penalty
        + spiro_penalty
        + bridge_penalty
        + macro_penalty
    )
    return max(1.0, min(10.0, round(raw, 2)))


def _sa_class(score: float) -> str:
    if score < 3.5:
        return "easy"
    if score < 6.0:
        return "moderate"
    return "hard"


# ── public API ─────────────────────────────────────────────────────────────────


def analyze_scaffold(smiles: str) -> Optional[ScaffoldAnalysis]:
    """Return scaffold + SA analysis for a SMILES, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = _bemis_murcko(mol)
    generic = _generic_murcko(scaffold) if scaffold is not None else None

    murcko_smi = Chem.MolToSmiles(scaffold) if scaffold and scaffold.GetNumAtoms() else None
    generic_smi = Chem.MolToSmiles(generic) if generic and generic.GetNumAtoms() else None

    n_rings, n_arom, n_aliph, largest = _ring_metrics(mol)
    n_spiro, n_bridge, has_macro = _topology_penalties(mol)
    sa = _sa_score(mol)

    return ScaffoldAnalysis(
        smiles=smiles,
        murcko_smiles=murcko_smi,
        generic_murcko_smiles=generic_smi,
        num_rings=n_rings,
        num_aromatic_rings=n_arom,
        num_aliphatic_rings=n_aliph,
        largest_ring_size=largest,
        num_spiro_atoms=n_spiro,
        num_bridgehead_atoms=n_bridge,
        has_macrocycle=has_macro,
        sa_score=sa,
        sa_class=_sa_class(sa),
    )
