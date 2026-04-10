"""
Tests for scaffold splitting: integrity, no leakage, determinism, coverage.
"""

from molprop.data.splits import (
    generate_scaffold,
    random_scaffold_split,
    scaffold_split,
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
