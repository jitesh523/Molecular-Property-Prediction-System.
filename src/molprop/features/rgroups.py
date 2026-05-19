"""
R-group decomposition.

Given a core scaffold (SMARTS or SMILES) and a list of analogue SMILES,
extracts the R-group at each attachment point of the core for every match.
Wraps RDKit's `rdRGroupDecomposition.RGroupDecompose` with safe defaults.

Returns:
    - Per-molecule R-group SMILES for R1, R2, ... (None if molecule does not match)
    - Aggregated statistics (unique R-groups per position, coverage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdRGroupDecomposition


@dataclass
class RGroupRow:
    smiles: str
    matched: bool
    core_smarts: Optional[str]
    r_groups: dict[str, str]  # e.g. {"R1": "CC", "R2": "O"}


@dataclass
class RGroupDecomposition:
    core: str
    n_input: int
    n_matched: int
    n_unmatched: int
    rgroup_labels: list[str]  # e.g. ["R1", "R2", "R3"]
    unique_per_label: dict[str, list[str]]
    rows: list[RGroupRow]

    def to_dict(self) -> dict:
        return {
            "core": self.core,
            "n_input": self.n_input,
            "n_matched": self.n_matched,
            "n_unmatched": self.n_unmatched,
            "rgroup_labels": self.rgroup_labels,
            "unique_per_label": self.unique_per_label,
            "rows": [
                {
                    "smiles": r.smiles,
                    "matched": r.matched,
                    "core_smarts": r.core_smarts,
                    "r_groups": r.r_groups,
                }
                for r in self.rows
            ],
        }


def _to_mol(smiles_or_smarts: str) -> Optional[Chem.Mol]:
    m = Chem.MolFromSmiles(smiles_or_smarts)
    if m is not None:
        return m
    return Chem.MolFromSmarts(smiles_or_smarts)


def decompose_rgroups(core: str, smiles_list: list[str]) -> Optional[RGroupDecomposition]:
    """
    Run R-group decomposition against a core.

    Returns ``None`` if the core is invalid.
    """
    core_mol = _to_mol(core)
    if core_mol is None:
        return None

    mols: list[Chem.Mol] = []
    valid_smiles: list[str] = []
    invalid_indices: list[int] = []
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            invalid_indices.append(i)
            continue
        mols.append(m)
        valid_smiles.append(smi)

    if not mols:
        return RGroupDecomposition(
            core=core,
            n_input=len(smiles_list),
            n_matched=0,
            n_unmatched=len(smiles_list),
            rgroup_labels=[],
            unique_per_label={},
            rows=[
                RGroupRow(smiles=s, matched=False, core_smarts=None, r_groups={})
                for s in smiles_list
            ],
        )

    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    params.removeHydrogensPostMatch = True
    params.alignment = rdRGroupDecomposition.RGroupCoreAlignment.MCS
    params.matchingStrategy = rdRGroupDecomposition.RGroupMatching.GreedyChunks

    decomp = rdRGroupDecomposition.RGroupDecomposition([core_mol], params)
    matched_flags: list[bool] = [False] * len(mols)
    for i, m in enumerate(mols):
        try:
            res = decomp.Add(m)
            matched_flags[i] = res >= 0
        except Exception:
            matched_flags[i] = False

    try:
        decomp.Process()
        groups = decomp.GetRGroupsAsRows()  # list[dict[label -> Mol]]
    except Exception:
        groups = []

    # Build per-molecule rows
    rows: list[RGroupRow] = []
    g_idx = 0
    labels_seen: set[str] = set()
    unique_per_label: dict[str, set[str]] = {}

    for smi, ok in zip(valid_smiles, matched_flags, strict=False):
        if not ok or g_idx >= len(groups):
            rows.append(RGroupRow(smiles=smi, matched=False, core_smarts=None, r_groups={}))
            continue
        g = groups[g_idx]
        g_idx += 1
        r_dict: dict[str, str] = {}
        core_smarts: Optional[str] = None
        for label, mol in g.items():
            if label.lower() == "core":
                core_smarts = Chem.MolToSmiles(mol)
                continue
            labels_seen.add(label)
            rs = Chem.MolToSmiles(mol)
            r_dict[label] = rs
            unique_per_label.setdefault(label, set()).add(rs)
        rows.append(
            RGroupRow(
                smiles=smi,
                matched=True,
                core_smarts=core_smarts,
                r_groups=r_dict,
            )
        )

    # Include invalid SMILES as unmatched rows
    for idx in invalid_indices:
        rows.insert(
            idx,
            RGroupRow(
                smiles=smiles_list[idx],
                matched=False,
                core_smarts=None,
                r_groups={},
            ),
        )

    n_matched = sum(1 for r in rows if r.matched)
    return RGroupDecomposition(
        core=core,
        n_input=len(smiles_list),
        n_matched=n_matched,
        n_unmatched=len(smiles_list) - n_matched,
        rgroup_labels=sorted(labels_seen, key=lambda x: (len(x), x)),
        unique_per_label={k: sorted(v) for k, v in unique_per_label.items()},
        rows=rows,
    )


# Touch AllChem to silence unused-import linters when this module is imported
# without using the rdkit-AllChem features directly (helps `ruff`).
_ = AllChem
