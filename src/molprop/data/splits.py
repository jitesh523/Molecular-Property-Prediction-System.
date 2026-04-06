import logging
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

log = logging.getLogger(__name__)


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)


def scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Perform a scaffold split on a list of SMILES.
    
    Splits are performed by grouping molecules by their Bemis-Murcko scaffold
    and assigning whole scaffold groups to each set.
    """
    np.testing.assert_almost_equal(frac_train + frac_val + frac_test, 1.0)

    # 1. Map each molecule index to its scaffold
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    # 2. Sort scaffold groups by size (descending)
    # This ensures that large scaffold families are handled consistently.
    # Note: Some versions of this algorithm use randomized sorting within size groups.
    sorted_scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    train_inds, val_inds, test_inds = [], [], []
    train_cutoff = frac_train * len(smiles_list)
    val_cutoff = (frac_train + frac_val) * len(smiles_list)

    # 3. Distribute scaffold sets into splits
    for scaffold_set in sorted_scaffold_sets:
        if len(train_inds) + len(scaffold_set) <= train_cutoff:
            train_inds.extend(scaffold_set)
        elif len(train_inds) + len(val_inds) + len(scaffold_set) <= val_cutoff:
            val_inds.extend(scaffold_set)
        else:
            test_inds.extend(scaffold_set)

    return train_inds, val_inds, test_inds


def random_scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Perform a randomized scaffold split.
    Similar to scaffold_split but shuffles the scaffold groups first.
    """
    np.testing.assert_almost_equal(frac_train + frac_val + frac_test, 1.0)
    random.seed(seed)

    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    scaffold_sets = list(scaffolds.values())
    random.shuffle(scaffold_sets)

    train_inds, val_inds, test_inds = [], [], []
    train_cutoff = frac_train * len(smiles_list)
    val_cutoff = (frac_train + frac_val) * len(smiles_list)

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) <= train_cutoff:
            train_inds.extend(scaffold_set)
        elif len(train_inds) + len(val_inds) + len(scaffold_set) <= val_cutoff:
            val_inds.extend(scaffold_set)
        else:
            test_inds.extend(scaffold_set)

    return train_inds, val_inds, test_inds
