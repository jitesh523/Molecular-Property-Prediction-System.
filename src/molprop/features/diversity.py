"""
MaxMin diverse-subset selection.

The classic MaxMin (a.k.a. "farthest-first traversal") algorithm:

1. Pick a seed molecule (defaulting to index 0).
2. Repeatedly pick the molecule whose **minimum** distance to the already-
   selected set is **maximum** — i.e. the one farthest from anything you've
   already chosen.
3. Stop when ``n_pick`` molecules have been chosen.

Distance is ``1 - Tanimoto(Morgan/MACCS fingerprint)``.

This guarantees that every newly selected molecule is at least as far from
its nearest neighbour in the picked set as every unpicked one — making it
the canonical algorithm for picking diverse screening subsets / training
sets, and far more useful than random sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys


@dataclass
class DiversitySelectionResult:
    n_input: int
    n_valid: int
    n_picked: int
    method: str
    seed_index: int
    picked_indices: list[int] = field(default_factory=list)
    picked_smiles: list[str] = field(default_factory=list)
    min_distances: list[float] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_valid": self.n_valid,
            "n_picked": self.n_picked,
            "method": self.method,
            "seed_index": self.seed_index,
            "picked_indices": list(self.picked_indices),
            "picked_smiles": list(self.picked_smiles),
            "min_distances": list(self.min_distances),
            "unmatched": list(self.unmatched),
        }


def _fingerprint(mol, method: str, radius: int, n_bits: int):
    if method == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def maxmin_select(
    smiles_list: list[str],
    n_pick: int,
    method: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    seed_index: Optional[int] = None,
) -> DiversitySelectionResult:
    """
    Greedy MaxMin diverse-subset picker.

    Parameters
    ----------
    smiles_list
        Candidate SMILES.
    n_pick
        Number of molecules to select. Clamped to ``n_valid``.
    method
        ``"morgan"`` (ECFP4-style) or ``"maccs"``.
    radius, n_bits
        Morgan parameters (ignored for MACCS).
    seed_index
        Index of the initial pick. Defaults to the first valid SMILES.

    Returns
    -------
    DiversitySelectionResult with the picked indices (into the original list),
    canonical SMILES, and the minimum distance to the previously-picked set
    at the moment each molecule was chosen (the seed gets distance ``1.0``).
    """
    n = len(smiles_list)
    fps: list = [None] * n
    canonical: list[str] = [""] * n
    unmatched: list[str] = []
    valid_idx: list[int] = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            unmatched.append(smi)
            continue
        canonical[i] = Chem.MolToSmiles(mol)
        fps[i] = _fingerprint(mol, method, radius, n_bits)
        valid_idx.append(i)

    n_valid = len(valid_idx)
    n_pick_clamped = max(0, min(n_pick, n_valid))
    if n_pick_clamped == 0:
        return DiversitySelectionResult(
            n_input=n,
            n_valid=n_valid,
            n_picked=0,
            method=method,
            seed_index=-1,
            unmatched=unmatched,
        )

    # Seed
    if seed_index is None or seed_index < 0 or seed_index >= n or fps[seed_index] is None:
        seed = valid_idx[0]
    else:
        seed = seed_index

    picked: list[int] = [seed]
    min_dist: list[float] = [1.0]  # seed has no prior; record sentinel distance

    # Precompute distance vector: distance from every valid mol to the
    # current "picked set", as min over picked.
    # Use 1 - Tanimoto for distance.
    dist_to_picked = np.zeros(n, dtype=float)
    for i in range(n):
        if fps[i] is None:
            dist_to_picked[i] = -np.inf
            continue
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[seed])
        dist_to_picked[i] = 1.0 - sim
    dist_to_picked[seed] = -np.inf  # never re-pick the seed

    while len(picked) < n_pick_clamped:
        # Pick the unpicked molecule with max distance to nearest picked
        next_idx = int(np.argmax(dist_to_picked))
        if not np.isfinite(dist_to_picked[next_idx]):
            break
        picked.append(next_idx)
        min_dist.append(float(dist_to_picked[next_idx]))
        # Update: each remaining row's distance to picked set is min(prev, dist to new)
        new_fp = fps[next_idx]
        for i in range(n):
            if dist_to_picked[i] == -np.inf:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], new_fp)
            d = 1.0 - sim
            if d < dist_to_picked[i]:
                dist_to_picked[i] = d
        dist_to_picked[next_idx] = -np.inf

    return DiversitySelectionResult(
        n_input=n,
        n_valid=n_valid,
        n_picked=len(picked),
        method=method,
        seed_index=seed,
        picked_indices=picked,
        picked_smiles=[canonical[i] for i in picked],
        min_distances=[round(d, 6) for d in min_dist],
        unmatched=unmatched,
    )
