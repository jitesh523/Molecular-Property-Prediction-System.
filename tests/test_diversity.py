"""Tests for MaxMin diverse-subset selection (v2.45)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from molprop.features.diversity import maxmin_select
from molprop.serving.api import app


class TestMaxMinCore:
    def test_n_pick_clamped_to_valid(self):
        r = maxmin_select(["c1ccccc1", "CCO"], n_pick=10)
        assert r.n_picked == 2

    def test_seed_is_first_when_unspecified(self):
        r = maxmin_select(["CCO", "c1ccccc1", "CCN"], n_pick=2)
        assert r.seed_index == 0
        assert r.picked_indices[0] == 0

    def test_explicit_seed_honoured(self):
        r = maxmin_select(["CCO", "c1ccccc1", "CCN"], n_pick=2, seed_index=1)
        assert r.seed_index == 1
        assert r.picked_indices[0] == 1

    def test_picked_indices_unique(self):
        # 5 analogues, pick 3 — must not repeat indices
        smiles = ["c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C)cc1", "CCCCCCCC", "C1CCCCC1"]
        r = maxmin_select(smiles, n_pick=3)
        assert len(set(r.picked_indices)) == 3

    def test_invalid_smiles_skipped(self):
        r = maxmin_select(["XYZ", "CCO", "c1ccccc1"], n_pick=2)
        assert "XYZ" in r.unmatched
        assert r.n_valid == 2
        # Invalid index 0 must not appear in picks
        assert 0 not in r.picked_indices

    def test_min_distances_monotonic_non_increasing_after_seed(self):
        """
        Classical MaxMin property: the distance of each newly-picked molecule
        to the picked-set so far is monotonically non-increasing as we add
        more picks (you can never improve the worst distance by picking more).
        """
        smiles = [
            "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C)cc1",
            "CCCCCCCC", "C1CCCCC1", "Nc1ccccc1O",
        ]
        r = maxmin_select(smiles, n_pick=5)
        # Skip the seed (index 0 sentinel == 1.0). Remaining should be non-increasing.
        dists = r.min_distances[1:]
        for a, b in zip(dists, dists[1:]):
            assert b <= a + 1e-9


class TestDiversityEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_endpoint_round_trip(self):
        r = self.client.post(
            "/diversity/select",
            json={
                "smiles_list": [
                    "c1ccc(O)cc1", "c1ccc(N)cc1", "CCCCCCCC", "C1CCCCC1",
                ],
                "n_pick": 2,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert d["n_picked"] == 2
        assert len(d["picked_smiles"]) == 2

    def test_endpoint_n_pick_too_large_validated_at_run_time(self):
        # Exceeding valid count is clamped at runtime (still 200 OK)
        r = self.client.post(
            "/diversity/select",
            json={"smiles_list": ["CCO", "c1ccccc1"], "n_pick": 5},
        )
        assert r.status_code == 200
        assert r.json()["n_picked"] == 2
