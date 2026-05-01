import logging
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedShuffleSplit

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


def scaffold_kfold(
    smiles_list: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    K-fold scaffold cross-validation.

    Groups molecules by Bemis-Murcko scaffold, shuffles scaffold groups,
    then distributes them across k folds. Returns a list of
    (train_indices, val_indices) tuples.

    This gives more robust performance estimates on small datasets
    like FreeSolv/ESOL compared to a single scaffold split.
    """
    random.seed(seed)

    # Group by scaffold
    scaffolds = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    scaffold_sets = list(scaffolds.values())
    random.shuffle(scaffold_sets)

    # Distribute scaffold groups round-robin into k folds
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for i, scaffold_set in enumerate(scaffold_sets):
        folds[i % n_folds].extend(scaffold_set)

    # Generate (train, val) for each fold
    splits = []
    for fold_idx in range(n_folds):
        val_inds = folds[fold_idx]
        train_inds = []
        for j in range(n_folds):
            if j != fold_idx:
                train_inds.extend(folds[j])
        splits.append((train_inds, val_inds))

    return splits


def stratified_split(
    labels: List[int],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified random split that preserves class balance across train/val/test.

    Useful for imbalanced classification datasets (e.g., BBBP, HIV) where
    random splitting can lead to skewed label distributions in small sets.

    Args:
        labels: Integer class labels for each molecule.
        frac_train: Fraction of data for training.
        frac_val: Fraction of data for validation.
        frac_test: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    np.testing.assert_almost_equal(frac_train + frac_val + frac_test, 1.0)

    labels_arr = np.array(labels)
    all_indices = np.arange(len(labels_arr))

    # First split: hold out test set
    test_frac = frac_test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(all_indices, labels_arr))

    # Second split: split remaining into train and val
    val_frac_of_remaining = frac_val / (frac_train + frac_val)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_remaining, random_state=seed)
    train_idx, val_idx = next(sss_val.split(train_val_idx, labels_arr[train_val_idx]))

    return (
        train_val_idx[train_idx].tolist(),
        train_val_idx[val_idx].tolist(),
        test_idx.tolist(),
    )


def temporal_split(
    df: pd.DataFrame,
    time_col: str,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Perform a temporal split on a dataframe.

    Sorts by time_col and allocates the earliest molecules to train
    and the latest to test. This simulates the real-world scenario
    of predicting future assay results.
    """
    np.testing.assert_almost_equal(frac_train + frac_val + frac_test, 1.0)

    # Sort indices by time column
    sorted_df = df.sort_values(by=time_col)
    indices = sorted_df.index.tolist()

    n = len(indices)
    train_cutoff = int(frac_train * n)
    val_cutoff = int((frac_train + frac_val) * n)

    train_inds = indices[:train_cutoff]
    val_inds = indices[train_cutoff:val_cutoff]
    test_inds = indices[val_cutoff:]

    return train_inds, val_inds, test_inds
