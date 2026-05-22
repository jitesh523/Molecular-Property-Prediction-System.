"""Tests for scaffold clustering (v2.42)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from molprop.features.scaffold_cluster import cluster_by_scaffold
from molprop.serving.api import app


class TestScaffoldCluster:
    def test_empty(self):
        r = cluster_by_scaffold([])
        assert r.n_input == 0
        assert r.n_valid == 0
        assert r.n_clusters == 0

    def test_benzene_analogues_collapse_to_single_cluster(self):
        analogues = [
            "c1ccc(O)cc1",
            "c1ccc(N)cc1",
            "c1ccc(C)cc1",
            "c1ccc(F)cc1",
            "c1ccc(Cl)cc1",
        ]
        r = cluster_by_scaffold(analogues, generic=False)
        assert r.n_valid == 5
        # All five share the benzene scaffold
        assert r.n_clusters == 1
        assert r.clusters[0].size == 5
        assert "c1ccccc1" == r.clusters[0].scaffold_smiles

    def test_invalid_smiles_reported_unmatched(self):
        r = cluster_by_scaffold(["XYZ", "c1ccccc1O", "c1ccccc1N"])
        assert "XYZ" in r.unmatched
        assert r.n_valid == 2

    def test_generic_collapses_isosteres(self):
        # Pyridine and benzene differ by one N→C; generic framework should merge them
        r = cluster_by_scaffold(["c1ccncc1", "c1ccccc1"], generic=True)
        assert r.n_clusters == 1
        assert r.clusters[0].size == 2

    def test_min_cluster_size_filter(self):
        # 3 benzenes + 1 lone naphthalene → only benzene cluster survives at min=2
        r = cluster_by_scaffold(
            ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccc2ccccc2c1"],
            min_cluster_size=2,
        )
        assert r.n_clusters == 1
        assert r.clusters[0].size == 3

    def test_clusters_sorted_by_size_desc(self):
        smiles = (
            ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C"]
            + ["c1ccc2ccccc2c1", "c1ccc2ccccc2c1Br"]
        )
        r = cluster_by_scaffold(smiles)
        sizes = [c.size for c in r.clusters]
        assert sizes == sorted(sizes, reverse=True)


class TestScaffoldClusterEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_endpoint_round_trip(self):
        r = self.client.post(
            "/scaffold/cluster",
            json={"smiles_list": ["c1ccccc1O", "c1ccccc1N", "c1ccc2ccccc2c1"]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["n_valid"] == 3
        assert data["n_clusters"] == 2

    def test_endpoint_generic_flag(self):
        r = self.client.post(
            "/scaffold/cluster",
            json={"smiles_list": ["c1ccncc1", "c1ccccc1"], "generic": True},
        )
        assert r.status_code == 200
        assert r.json()["n_clusters"] == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
