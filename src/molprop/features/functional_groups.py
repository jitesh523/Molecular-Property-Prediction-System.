"""
Functional group detection via SMARTS pattern matching.

Provides a curated catalog of ~30 common functional groups important in
medicinal chemistry. For each detected group we return:
  - the group name and SMARTS pattern
  - the matching atom indices (so the UI can highlight them on a 2D depiction)
  - a count of occurrences

Hetero-aromatic and aliphatic groups are kept distinct so the depiction is
chemically meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem

# (name, SMARTS, category)
FUNCTIONAL_GROUPS: list[tuple[str, str, str]] = [
    # Carbonyl-based
    ("Carboxylic acid", "[CX3](=O)[OX2H1]", "Carbonyl"),
    ("Ester", "[CX3](=O)[OX2][CX4,c]", "Carbonyl"),
    ("Amide", "[NX3][CX3](=O)[#6]", "Carbonyl"),
    ("Aldehyde", "[CX3H1](=O)[#6]", "Carbonyl"),
    ("Ketone", "[#6][CX3](=O)[#6]", "Carbonyl"),
    ("Anhydride", "[CX3](=O)[OX2][CX3](=O)", "Carbonyl"),
    ("Acid chloride", "[CX3](=O)[Cl]", "Carbonyl"),
    ("Carbamate", "[NX3][CX3](=O)[OX2][#6]", "Carbonyl"),
    ("Urea", "[NX3][CX3](=O)[NX3]", "Carbonyl"),
    # Nitrogen
    ("Primary amine", "[NX3;H2;!$(NC=O)]", "Amine"),
    ("Secondary amine", "[NX3;H1;!$(NC=O)]([#6])[#6]", "Amine"),
    ("Tertiary amine", "[NX3;H0;!$(NC=O)]([#6])([#6])[#6]", "Amine"),
    ("Quaternary ammonium", "[NX4+]", "Amine"),
    ("Nitrile", "[CX2]#[NX1]", "Nitrogen"),
    ("Nitro", "[NX3+](=O)[O-]", "Nitrogen"),
    ("Imine", "[CX3]=[NX2][#6,H]", "Nitrogen"),
    ("Hydrazine", "[NX3][NX3]", "Nitrogen"),
    ("Azo", "[#6][NX2]=[NX2][#6]", "Nitrogen"),
    ("Azide", "[NX1]=[NX2+]=[NX1-]", "Nitrogen"),
    # Oxygen
    ("Alcohol", "[OX2H][CX4;!$(C=O)]", "Oxygen"),
    ("Phenol", "[OX2H][c]", "Oxygen"),
    ("Ether", "[OD2]([#6])[#6]", "Oxygen"),
    ("Peroxide", "[OX2][OX2]", "Oxygen"),
    # Sulfur / phosphorus
    ("Thiol", "[SX2H]", "Sulfur"),
    ("Thioether", "[SD2]([#6])[#6]", "Sulfur"),
    ("Sulfonamide", "[SX4](=O)(=O)[NX3]", "Sulfur"),
    ("Sulfonic acid", "[SX4](=O)(=O)[OX2H]", "Sulfur"),
    ("Sulfone", "[SX4](=O)(=O)([#6])[#6]", "Sulfur"),
    ("Phosphate", "[PX4](=O)([OX2])([OX2])[OX2]", "Phosphorus"),
    # Halogens
    ("Aryl halide", "[F,Cl,Br,I][c]", "Halogen"),
    ("Alkyl halide", "[F,Cl,Br,I][CX4]", "Halogen"),
    # Aromatic / heteroaromatic
    ("Benzene", "c1ccccc1", "Aromatic"),
    ("Pyridine", "n1ccccc1", "Heteroaromatic"),
    ("Pyrrole", "[nH]1cccc1", "Heteroaromatic"),
    ("Imidazole", "c1ncc[nH]1", "Heteroaromatic"),
    ("Furan", "o1cccc1", "Heteroaromatic"),
    ("Thiophene", "s1cccc1", "Heteroaromatic"),
    ("Indole", "c1ccc2[nH]ccc2c1", "Heteroaromatic"),
    # Other
    ("Epoxide", "C1OC1", "Cyclic"),
    ("Aziridine", "C1NC1", "Cyclic"),
    ("Alkene", "[CX3]=[CX3]", "Aliphatic"),
    ("Alkyne", "[CX2]#[CX2]", "Aliphatic"),
]


@dataclass
class FunctionalGroupHit:
    name: str
    smarts: str
    category: str
    count: int
    atom_indices: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class FunctionalGroupAnalysis:
    smiles: str
    canonical_smiles: Optional[str]
    num_heavy_atoms: int
    num_atoms_total: int
    num_groups_found: int
    groups: list[FunctionalGroupHit]
    categories: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "num_heavy_atoms": self.num_heavy_atoms,
            "num_atoms_total": self.num_atoms_total,
            "num_groups_found": self.num_groups_found,
            "groups": [g.to_dict() for g in self.groups],
            "categories": self.categories,
        }


def detect_functional_groups(smiles: str) -> Optional[FunctionalGroupAnalysis]:
    """Return all functional groups detected in the molecule.

    Returns ``None`` for invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    hits: list[FunctionalGroupHit] = []
    for name, smarts, category in FUNCTIONAL_GROUPS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt, uniquify=True)
        if not matches:
            continue
        hits.append(
            FunctionalGroupHit(
                name=name,
                smarts=smarts,
                category=category,
                count=len(matches),
                atom_indices=[list(m) for m in matches],
            )
        )

    categories: dict[str, int] = {}
    for h in hits:
        categories[h.category] = categories.get(h.category, 0) + h.count

    return FunctionalGroupAnalysis(
        smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_atoms_total=mol.GetNumAtoms(),
        num_groups_found=len(hits),
        groups=hits,
        categories=categories,
    )
