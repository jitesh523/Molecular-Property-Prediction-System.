"""
Tests for GNN model architectures: GCN, GAT, MPNN, GIN.

Verifies forward passes, encode() methods, mc_dropout inference,
and output shapes for both single-molecule and batch inputs.
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from molprop.features.graphs import smiles_to_graph
from molprop.models.gnn_gat import GATModel
from molprop.models.gnn_gcn import GCNModel
from molprop.models.gnn_gin import GINModel
from molprop.models.gnn_mpnn import MPNNModel

BENZENE = "c1ccccc1"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"

IN_DIM = 9
HIDDEN_DIM = 32
OUT_DIM = 1
EDGE_DIM = 3


def _make_graph(smiles: str) -> Data:
    g = smiles_to_graph(smiles)
    assert g is not None
    g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    return g


def _make_batch(smiles_list) -> Batch:
    graphs = [smiles_to_graph(s) for s in smiles_list]
    return Batch.from_data_list(graphs)


@pytest.fixture
def gcn():
    return GCNModel(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)


@pytest.fixture
def gat():
    return GATModel(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, heads=2)


@pytest.fixture
def mpnn():
    return MPNNModel(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, edge_dim=EDGE_DIM)


@pytest.fixture
def gin():
    return GINModel(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM)


class TestGCNModel:
    def test_forward_single(self, gcn):
        g = _make_graph(BENZENE)
        out = gcn(g)
        assert out.shape == (1, OUT_DIM)

    def test_forward_batch(self, gcn):
        batch = _make_batch([BENZENE, ASPIRIN])
        out = gcn(batch)
        assert out.shape == (2, OUT_DIM)

    def test_encode_shape(self, gcn):
        g = _make_graph(BENZENE)
        emb = gcn.encode(g)
        assert emb.shape == (1, HIDDEN_DIM)

    def test_mc_dropout(self, gcn):
        g = _make_graph(BENZENE)
        out1 = gcn(g, mc_dropout=True)
        assert out1.shape == (1, OUT_DIM)

    def test_eval_deterministic(self, gcn):
        gcn.eval()
        g = _make_graph(BENZENE)
        with torch.no_grad():
            out1 = gcn(g)
            out2 = gcn(g)
        assert torch.allclose(out1, out2)


class TestGATModel:
    def test_forward_single(self, gat):
        g = _make_graph(BENZENE)
        out = gat(g)
        assert out.shape == (1, OUT_DIM)

    def test_forward_batch(self, gat):
        batch = _make_batch([BENZENE, ASPIRIN])
        out = gat(batch)
        assert out.shape == (2, OUT_DIM)

    def test_encode_shape(self, gat):
        g = _make_graph(BENZENE)
        emb = gat.encode(g)
        assert emb.shape == (1, HIDDEN_DIM)

    def test_mc_dropout(self, gat):
        g = _make_graph(BENZENE)
        out = gat(g, mc_dropout=True)
        assert out.shape == (1, OUT_DIM)


class TestMPNNModel:
    def test_forward_single(self, mpnn):
        g = _make_graph(BENZENE)
        out = mpnn(g)
        assert out.shape == (1, OUT_DIM)

    def test_forward_batch(self, mpnn):
        batch = _make_batch([BENZENE, ASPIRIN])
        out = mpnn(batch)
        assert out.shape == (2, OUT_DIM)

    def test_encode_shape(self, mpnn):
        g = _make_graph(BENZENE)
        emb = mpnn.encode(g)
        assert emb.shape == (1, HIDDEN_DIM)

    def test_mc_dropout(self, mpnn):
        g = _make_graph(BENZENE)
        out = mpnn(g, mc_dropout=True)
        assert out.shape == (1, OUT_DIM)


class TestGINModel:
    def test_forward_single(self, gin):
        g = _make_graph(BENZENE)
        out = gin(g)
        assert out.shape == (1, OUT_DIM)

    def test_forward_batch(self, gin):
        batch = _make_batch([BENZENE, ASPIRIN])
        out = gin(batch)
        assert out.shape == (2, OUT_DIM)

    def test_encode_shape(self, gin):
        g = _make_graph(BENZENE)
        emb = gin.encode(g)
        assert emb.shape == (1, HIDDEN_DIM)

    def test_mc_dropout(self, gin):
        g = _make_graph(BENZENE)
        out = gin(g, mc_dropout=True)
        assert out.shape == (1, OUT_DIM)

    def test_eval_deterministic(self, gin):
        gin.eval()
        g = _make_graph(BENZENE)
        with torch.no_grad():
            out1 = gin(g)
            out2 = gin(g)
        assert torch.allclose(out1, out2)
