"""Tests for /similarity/matrix (v2.44)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from molprop.serving.api import app


class TestSimilarityMatrix:
    def setup_method(self):
        self.client = TestClient(app)

    def test_basic_morgan_matrix(self):
        r = self.client.post(
            "/similarity/matrix",
            json={"smiles_list": ["c1ccccc1", "c1ccc(O)cc1", "CCCCCCCC"]},
        )
        assert r.status_code == 200
        d = r.json()
        assert d["n"] == 3
        assert d["n_valid"] == 3
        assert d["method"] == "morgan"
        # 3×3 matrix
        assert len(d["matrix"]) == 3 and all(len(row) == 3 for row in d["matrix"])

    def test_diagonal_is_one(self):
        d = self.client.post(
            "/similarity/matrix",
            json={"smiles_list": ["CCO", "c1ccccc1"]},
        ).json()
        for i in range(d["n"]):
            assert d["matrix"][i][i] == 1.0

    def test_matrix_symmetric(self):
        d = self.client.post(
            "/similarity/matrix",
            json={"smiles_list": ["c1ccccc1", "c1ccc(N)cc1", "CCCCCCCC"]},
        ).json()
        n = d["n"]
        for i in range(n):
            for j in range(n):
                assert d["matrix"][i][j] == d["matrix"][j][i]

    def test_similar_pair_higher_than_unrelated(self):
        """Phenol/aniline (both ortho-substituted benzene) should be more similar
        to each other than either is to a long aliphatic chain."""
        d = self.client.post(
            "/similarity/matrix",
            json={"smiles_list": ["c1ccc(O)cc1", "c1ccc(N)cc1", "CCCCCCCC"]},
        ).json()
        m = d["matrix"]
        assert m[0][1] > m[0][2]
        assert m[0][1] > m[1][2]

    def test_invalid_smiles_zero_row(self):
        d = self.client.post(
            "/similarity/matrix",
            json={"smiles_list": ["XYZ", "c1ccccc1", "c1ccc(O)cc1"]},
        ).json()
        assert d["valid"] == [False, True, True]
        # Row/col 0 (invalid) is all zeros (incl. diagonal)
        assert d["matrix"][0] == [0.0, 0.0, 0.0]

    def test_maccs_method(self):
        d = self.client.post(
            "/similarity/matrix",
            json={"smiles_list": ["CCO", "c1ccccc1"], "method": "maccs"},
        ).json()
        assert d["method"] == "maccs"
        assert d["n"] == 2

    def test_nearest_neighbours(self):
        d = self.client.post(
            "/similarity/matrix",
            json={"smiles_list": ["c1ccc(O)cc1", "c1ccc(N)cc1", "CCCCCCCC"]},
        ).json()
        # Phenol's nearest neighbour should be aniline (j=1), not octane
        nn0 = [p for p in d["nearest_neighbours"] if p["i"] == 0][0]
        assert nn0["j"] == 1
