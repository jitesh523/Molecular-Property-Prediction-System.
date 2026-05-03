"""
Tests for scaffold splitting: integrity, no leakage, determinism, coverage.
"""

import pandas as pd
import pytest

from molprop.data.splits import (
    generate_scaffold,
    random_scaffold_split,
    scaffold_kfold,
    scaffold_split,
    stratified_split,
    temporal_split,
)

# Diverse SMILES that produce different Bemis-Murcko scaffolds
SAMPLE_SMILES = [
    "c1ccccc1",  # benzene
    "c1ccc(O)cc1",  # phenol
    "c1ccc(N)cc1",  # aniline
    "c1ccc2ccccc2c1",  # naphthalene
    "c1ccc2cc3ccccc3cc2c1",  # anthracene
    "C1CCCCC1",  # cyclohexane
    "C1CCC(CC1)O",  # cyclohexanol
    "CC(=O)O",  # acetic acid
    "CCO",  # ethanol
    "CCCC",  # butane
    "c1ccncc1",  # pyridine
    "c1ccc(cc1)c1ccccc1",  # biphenyl
    "C1CC1",  # cyclopropane
    "c1ccc(Cl)cc1",  # chlorobenzene
    "c1ccc(F)cc1",  # fluorobenzene
    "OC(=O)c1ccccc1",  # benzoic acid
    "CC(=O)Nc1ccccc1",  # acetanilide
    "c1cnc2ccccc2n1",  # quinazoline
    "c1ccc(-c2ncccc2)cc1",  # 2-phenylpyridine
    "CC(C)CC",  # isopentane
]


class TestGenerateScaffold:
    def test_valid_smiles(self):
        scaffold = generate_scaffold("c1ccccc1")
        assert scaffold != ""
        assert isinstance(scaffold, str)

    def test_invalid_smiles(self):
        scaffold = generate_scaffold("INVALID_SMILES")
        assert scaffold == ""

    def test_same_scaffold_for_derivatives(self):
        """Phenol and aniline share the benzene scaffold."""
        s1 = generate_scaffold("c1ccc(O)cc1")
        s2 = generate_scaffold("c1ccc(N)cc1")
        assert s1 == s2

    def test_different_scaffolds(self):
        """Benzene and naphthalene have different scaffolds."""
        s1 = generate_scaffold("c1ccccc1")
        s2 = generate_scaffold("c1ccc2ccccc2c1")
        assert s1 != s2


class TestScaffoldSplit:
    def test_no_index_overlap(self):
        """Train, val, test indices must be mutually exclusive."""
        train, val, test = scaffold_split(SAMPLE_SMILES)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_full_coverage(self):
        """Every molecule index must appear in exactly one split."""
        train, val, test = scaffold_split(SAMPLE_SMILES)
        all_indices = sorted(train + val + test)
        expected = list(range(len(SAMPLE_SMILES)))
        assert all_indices == expected

    def test_no_scaffold_leakage(self):
        """No Bemis-Murcko scaffold should appear in more than one split."""
        train, val, test = scaffold_split(SAMPLE_SMILES)

        def get_scaffolds(indices):
            return {generate_scaffold(SAMPLE_SMILES[i]) for i in indices}

        train_scaffolds = get_scaffolds(train)
        val_scaffolds = get_scaffolds(val)
        test_scaffolds = get_scaffolds(test)

        assert len(train_scaffolds & val_scaffolds) == 0, "Scaffold leak: train ∩ val"
        assert len(train_scaffolds & test_scaffolds) == 0, "Scaffold leak: train ∩ test"
        assert len(val_scaffolds & test_scaffolds) == 0, "Scaffold leak: val ∩ test"

    def test_approximate_ratios(self):
        """Split sizes should roughly match the specified fractions."""
        n = len(SAMPLE_SMILES)
        train, val, test = scaffold_split(SAMPLE_SMILES, 0.8, 0.1, 0.1)
        # Allow ±20% tolerance since scaffold groups are indivisible
        assert len(train) >= n * 0.5
        assert len(train) <= n * 1.0

    def test_determinism(self):
        """Same input → same output."""
        split1 = scaffold_split(SAMPLE_SMILES)
        split2 = scaffold_split(SAMPLE_SMILES)
        assert split1 == split2


class TestRandomScaffoldSplit:
    def test_no_index_overlap(self):
        train, val, test = random_scaffold_split(SAMPLE_SMILES, seed=42)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_full_coverage(self):
        train, val, test = random_scaffold_split(SAMPLE_SMILES, seed=42)
        all_indices = sorted(train + val + test)
        assert all_indices == list(range(len(SAMPLE_SMILES)))

    def test_seed_determinism(self):
        """Same seed → same split."""
        split1 = random_scaffold_split(SAMPLE_SMILES, seed=123)
        split2 = random_scaffold_split(SAMPLE_SMILES, seed=123)
        assert split1 == split2

    def test_different_seeds_differ(self):
        """Different seeds should (usually) produce different splits."""
        split1 = random_scaffold_split(SAMPLE_SMILES, seed=1)
        split2 = random_scaffold_split(SAMPLE_SMILES, seed=999)
        # At least one of train/val/test should differ
        assert split1 != split2


class TestStratifiedSplit:
    LABELS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    def test_no_index_overlap(self):
        train, val, test = stratified_split(self.LABELS)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_full_coverage(self):
        train, val, test = stratified_split(self.LABELS)
        all_indices = sorted(train + val + test)
        assert all_indices == list(range(len(self.LABELS)))

    def test_class_balance_train(self):
        """Training set should have roughly equal class ratio."""
        train, val, test = stratified_split(self.LABELS)
        train_labels = [self.LABELS[i] for i in train]
        ratio = sum(train_labels) / len(train_labels)
        assert 0.3 <= ratio <= 0.7

    def test_class_balance_test(self):
        """Test set should have roughly equal class ratio."""
        train, val, test = stratified_split(self.LABELS)
        test_labels = [self.LABELS[i] for i in test]
        ratio = sum(test_labels) / len(test_labels)
        assert 0.3 <= ratio <= 0.7

    def test_determinism(self):
        split1 = stratified_split(self.LABELS, seed=42)
        split2 = stratified_split(self.LABELS, seed=42)
        assert split1 == split2


class TestScaffoldKFold:
    def test_correct_fold_count(self):
        splits = scaffold_kfold(SAMPLE_SMILES, n_folds=5)
        assert len(splits) == 5

    def test_each_fold_is_tuple(self):
        splits = scaffold_kfold(SAMPLE_SMILES, n_folds=3)
        for train_idx, val_idx in splits:
            assert isinstance(train_idx, list)
            assert isinstance(val_idx, list)

    def test_no_overlap_within_fold(self):
        splits = scaffold_kfold(SAMPLE_SMILES, n_folds=5)
        for train_idx, val_idx in splits:
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_full_coverage_across_folds(self):
        """Union of all val sets must cover every molecule exactly once."""
        splits = scaffold_kfold(SAMPLE_SMILES, n_folds=5)
        all_val = [i for _, val in splits for i in val]
        assert sorted(all_val) == list(range(len(SAMPLE_SMILES)))

    def test_determinism(self):
        s1 = scaffold_kfold(SAMPLE_SMILES, n_folds=3, seed=42)
        s2 = scaffold_kfold(SAMPLE_SMILES, n_folds=3, seed=42)
        assert s1 == s2


class TestTemporalSplit:
    N = 30

    def _make_df(self, n=None):
        n = n or self.N
        return pd.DataFrame({"time": list(range(n)), "smiles": ["C"] * n})

    def test_no_index_overlap(self):
        df = self._make_df()
        train, val, test = temporal_split(df, "time")
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_full_coverage(self):
        df = self._make_df()
        train, val, test = temporal_split(df, "time")
        all_indices = sorted(train + val + test)
        assert all_indices == sorted(df.index.tolist())

    def test_chronological_order(self):
        """Train times must all precede test times."""
        df = self._make_df()
        train, val, test = temporal_split(df, "time")
        if train and test:
            max_train_time = df.loc[train, "time"].max()
            min_test_time = df.loc[test, "time"].min()
            assert max_train_time < min_test_time

    def test_approximate_ratios(self):
        df = self._make_df()
        train, val, test = temporal_split(df, "time", frac_train=0.8, frac_val=0.1, frac_test=0.1)
        n = len(df)
        assert len(train) == pytest.approx(n * 0.8, abs=1)
        assert len(val) == pytest.approx(n * 0.1, abs=1)
        assert len(test) == pytest.approx(n * 0.1, abs=1)

    def test_unsorted_input_still_sorted_by_time(self):
        """Input rows in reverse order — split should still be time-ordered."""
        df = pd.DataFrame({"time": list(range(20, 0, -1)), "smiles": ["C"] * 20})
        train, _, test = temporal_split(df, "time")
        if train and test:
            max_train_time = df.loc[train, "time"].max()
            min_test_time = df.loc[test, "time"].min()
            assert max_train_time < min_test_time

    def test_custom_fractions(self):
        df = self._make_df(60)
        train, val, test = temporal_split(df, "time", frac_train=0.7, frac_val=0.2, frac_test=0.1)
        assert len(train) + len(val) + len(test) == len(df)
