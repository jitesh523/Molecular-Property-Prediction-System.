"""
Tests for featurization modules: fingerprints, descriptors, graphs.
"""

import numpy as np

from molprop.features.descriptors import (
    batch_smiles_to_descriptors,
    get_descriptor_names,
    smiles_to_descriptors,
)
from molprop.features.fingerprints import (
    batch_smiles_to_maccs,
    batch_smiles_to_morgan,
    dice_similarity,
    smiles_to_maccs,
    smiles_to_morgan,
    tanimoto_similarity,
)
from molprop.features.graphs import batch_smiles_to_graphs, smiles_to_graph

VALID_SMILES = "c1ccccc1"  # benzene
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
INVALID_SMILES = "NOT_A_MOLECULE"


class TestMorganFingerprints:
    def test_valid_smiles_shape(self):
        fp = smiles_to_morgan(VALID_SMILES)
        assert fp is not None
        assert fp.shape == (2048,)

    def test_binary_values(self):
        """Morgan fingerprint should contain only 0s and 1s."""
        fp = smiles_to_morgan(VALID_SMILES)
        assert set(np.unique(fp)).issubset({0, 1})

    def test_invalid_smiles_returns_none(self):
        fp = smiles_to_morgan(INVALID_SMILES)
        assert fp is None

    def test_different_molecules_differ(self):
        fp1 = smiles_to_morgan("c1ccccc1")  # benzene
        fp2 = smiles_to_morgan("C1CCCCC1")  # cyclohexane
        assert not np.array_equal(fp1, fp2)

    def test_batch_conversion(self):
        smiles_list = [VALID_SMILES, ASPIRIN]
        batch_fp = batch_smiles_to_morgan(smiles_list)
        assert batch_fp.shape == (2, 2048)

    def test_custom_nbits(self):
        fp = smiles_to_morgan(VALID_SMILES, n_bits=1024)
        assert fp.shape == (1024,)


class TestMACCSFingerprints:
    def test_valid_smiles_shape(self):
        fp = smiles_to_maccs(VALID_SMILES)
        assert fp is not None
        assert fp.shape == (167,)

    def test_binary_values(self):
        fp = smiles_to_maccs(VALID_SMILES)
        assert set(np.unique(fp)).issubset({0, 1})

    def test_invalid_smiles_returns_none(self):
        fp = smiles_to_maccs(INVALID_SMILES)
        assert fp is None

    def test_different_molecules_differ(self):
        fp1 = smiles_to_maccs("c1ccccc1")
        fp2 = smiles_to_maccs(ASPIRIN)
        assert not np.array_equal(fp1, fp2)

    def test_batch_conversion(self):
        batch_fp = batch_smiles_to_maccs([VALID_SMILES, ASPIRIN])
        assert batch_fp.shape == (2, 167)


class TestDescriptors:
    def test_valid_smiles(self):
        desc = smiles_to_descriptors(VALID_SMILES)
        assert desc is not None
        assert isinstance(desc, np.ndarray)
        assert desc.dtype == np.float32

    def test_descriptor_count(self):
        """Should return exactly 18 descriptors."""
        desc = smiles_to_descriptors(VALID_SMILES)
        names = get_descriptor_names()
        assert len(desc) == len(names)
        assert len(names) == 18

    def test_no_nan_for_valid_smiles(self):
        desc = smiles_to_descriptors(VALID_SMILES)
        assert not np.any(np.isnan(desc))

    def test_invalid_smiles_returns_none(self):
        desc = smiles_to_descriptors(INVALID_SMILES)
        assert desc is None

    def test_batch_conversion(self):
        smiles_list = [VALID_SMILES, ASPIRIN]
        batch_desc = batch_smiles_to_descriptors(smiles_list)
        assert batch_desc.shape[0] == 2
        assert batch_desc.shape[1] == len(get_descriptor_names())

    def test_known_logp_range(self):
        """Benzene has a known logP around 1.6-1.7."""
        desc = smiles_to_descriptors(VALID_SMILES)
        logp = desc[0]  # MolLogP is the first descriptor
        assert 1.0 < logp < 2.5


class TestGraphConstruction:
    def test_valid_smiles(self):
        graph = smiles_to_graph(VALID_SMILES)
        assert graph is not None

    def test_node_features(self):
        """Benzene has 6 carbon atoms, so 6 nodes."""
        graph = smiles_to_graph(VALID_SMILES)
        assert graph.x.shape[0] == 6  # 6 atoms
        assert graph.x.shape[1] > 0  # some features per atom

    def test_edge_index_shape(self):
        graph = smiles_to_graph(VALID_SMILES)
        assert graph.edge_index.shape[0] == 2  # [2, num_edges]
        assert graph.edge_index.shape[1] > 0

    def test_smiles_attribute(self):
        graph = smiles_to_graph(VALID_SMILES)
        assert hasattr(graph, "smiles")
        assert graph.smiles == VALID_SMILES

    def test_with_label(self):
        graph = smiles_to_graph(VALID_SMILES, y=1.5)
        assert graph.y is not None
        assert float(graph.y) == 1.5

    def test_invalid_smiles_returns_none(self):
        graph = smiles_to_graph(INVALID_SMILES)
        assert graph is None

    def test_edge_symmetry(self):
        """Undirected graph: each bond creates 2 directed edges."""
        graph = smiles_to_graph(VALID_SMILES)
        # Benzene: 6 bonds → 12 directed edges
        assert graph.edge_index.shape[1] == 12


class TestTanimotoSimilarity:
    def test_identical_molecules(self):
        """Same molecule should have similarity 1.0."""
        sim = tanimoto_similarity(VALID_SMILES, VALID_SMILES)
        assert sim == 1.0

    def test_range(self):
        """Similarity must lie in [0, 1]."""
        sim = tanimoto_similarity(VALID_SMILES, ASPIRIN)
        assert sim is not None
        assert 0.0 <= sim <= 1.0

    def test_symmetry(self):
        """Tanimoto similarity is symmetric."""
        sim_ab = tanimoto_similarity(VALID_SMILES, ASPIRIN)
        sim_ba = tanimoto_similarity(ASPIRIN, VALID_SMILES)
        assert sim_ab == sim_ba

    def test_invalid_first_smiles_returns_none(self):
        sim = tanimoto_similarity(INVALID_SMILES, VALID_SMILES)
        assert sim is None

    def test_invalid_second_smiles_returns_none(self):
        sim = tanimoto_similarity(VALID_SMILES, INVALID_SMILES)
        assert sim is None

    def test_dissimilar_molecules(self):
        """Benzene and a large drug-like molecule should have low similarity."""
        sim = tanimoto_similarity("c1ccccc1", ASPIRIN)
        assert sim is not None
        assert sim < 0.8


class TestDiceSimilarity:
    def test_identical_molecules(self):
        """Same molecule must yield Dice similarity of 1.0."""
        assert dice_similarity(VALID_SMILES, VALID_SMILES) == 1.0

    def test_range(self):
        sim = dice_similarity(VALID_SMILES, ASPIRIN)
        assert sim is not None
        assert 0.0 <= sim <= 1.0

    def test_symmetry(self):
        assert dice_similarity(VALID_SMILES, ASPIRIN) == dice_similarity(ASPIRIN, VALID_SMILES)

    def test_invalid_first_smiles_returns_none(self):
        assert dice_similarity(INVALID_SMILES, VALID_SMILES) is None

    def test_invalid_second_smiles_returns_none(self):
        assert dice_similarity(VALID_SMILES, INVALID_SMILES) is None

    def test_dice_ge_tanimoto_for_different_sizes(self):
        """For two non-identical molecules Dice >= Tanimoto (provable algebraically)."""
        sim_dice = dice_similarity(VALID_SMILES, ASPIRIN)
        sim_tan = tanimoto_similarity(VALID_SMILES, ASPIRIN)
        assert sim_dice is not None and sim_tan is not None
        assert sim_dice >= sim_tan - 1e-6


class TestBatchSmilesToGraphs:
    SMILES = [VALID_SMILES, ASPIRIN, "CCO"]
    LABELS = [1.0, 0.0, 1.0]

    def test_length_matches_input(self):
        graphs = batch_smiles_to_graphs(self.SMILES)
        assert len(graphs) == len(self.SMILES)

    def test_all_valid_returns_data_objects(self):
        graphs = batch_smiles_to_graphs(self.SMILES)
        for g in graphs:
            assert g is not None

    def test_invalid_smiles_yields_none(self):
        graphs = batch_smiles_to_graphs([VALID_SMILES, INVALID_SMILES])
        assert graphs[0] is not None
        assert graphs[1] is None

    def test_labels_attached_to_y(self):
        graphs = batch_smiles_to_graphs(self.SMILES, labels=self.LABELS)
        for g, lbl in zip(graphs, self.LABELS, strict=False):
            assert g is not None
            assert g.y is not None
            assert float(g.y) == lbl

    def test_no_labels_y_is_none(self):
        graphs = batch_smiles_to_graphs(self.SMILES)
        for g in graphs:
            assert g.y is None

    def test_node_feature_dim(self):
        graphs = batch_smiles_to_graphs([VALID_SMILES])
        assert graphs[0].x.shape[1] == 9
