"""Tests for /standardize/batch (v2.43)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from molprop.serving.api import app


class TestStandardizeBatch:
    def setup_method(self):
        self.client = TestClient(app)

    def test_round_trip_all_valid(self):
        r = self.client.post(
            "/standardize/batch",
            json={"smiles_list": ["CCO", "c1ccccc1", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["n_input"] == 3
        assert data["n_valid"] == 3
        assert data["n_invalid"] == 0
        assert len(data["rows"]) == 3
        for row in data["rows"]:
            assert row["valid"] is True
            assert row["final"] is not None

    def test_invalid_rows_reported_not_raised(self):
        """Invalid SMILES are flagged ``valid=false`` rather than aborting the batch."""
        r = self.client.post(
            "/standardize/batch",
            json={"smiles_list": ["CCO", "XYZ123", "c1ccccc1"]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["n_input"] == 3
        assert data["n_valid"] == 2
        assert data["n_invalid"] == 1
        invalid_rows = [r for r in data["rows"] if not r["valid"]]
        assert len(invalid_rows) == 1
        assert invalid_rows[0]["input"] == "XYZ123"
        assert invalid_rows[0]["final"] is None

    def test_changed_counter_increments(self):
        """Non-canonical input should be reported as ``changed``."""
        r = self.client.post(
            "/standardize/batch",
            json={"smiles_list": ["OCC", "c1ccccc1"]},  # OCC → CCO
        )
        data = r.json()
        # At least the non-canonical one is reported changed
        assert data["n_changed"] >= 1

    def test_empty_list_rejected(self):
        r = self.client.post("/standardize/batch", json={"smiles_list": []})
        # min_length=1 → 422 from pydantic
        assert r.status_code == 422
