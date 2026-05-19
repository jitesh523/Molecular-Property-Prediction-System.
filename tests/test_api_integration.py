"""
Integration tests for the molprop FastAPI app using FastAPI's TestClient.

These tests exercise the public HTTP surface end-to-end without spinning up
a real server. They cover the cheminformatics endpoints introduced in
v2.5 – v2.19 (scaffold, functional groups, isomers, substructure, compare,
standardize, MCS, alerts, report, library CRUD + CSV import/export).

ML / heavy endpoints (predict, generate, ADMET, VAE) are skipped here to
keep the test suite light and CI-friendly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Import the app lazily so that module-level model loading errors don't
# crash test collection on machines without trained weights.
pytest.importorskip("rdkit", reason="RDKit required for API integration tests")


@pytest.fixture(scope="module")
def client():
    from molprop.serving.api import app

    return TestClient(app)


# ── Scaffold ─────────────────────────────────────────────────────────────────


def test_scaffold_aspirin(client):
    r = client.post("/scaffold", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert r.status_code == 200
    body = r.json()
    assert body["num_rings"] == 1
    assert body["num_aromatic_rings"] == 1
    assert 1.0 <= body["sa_score"] <= 10.0


def test_scaffold_invalid_smiles_returns_422(client):
    r = client.post("/scaffold", json={"smiles": "not a smiles"})
    assert r.status_code == 422


# ── Functional groups ───────────────────────────────────────────────────────


def test_functional_groups_aspirin(client):
    r = client.post("/functional_groups", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert r.status_code == 200
    body = r.json()
    names = {g["name"] for g in body["groups"]}
    assert "Carboxylic acid" in names
    assert "Ester" in names


# ── Isomers ─────────────────────────────────────────────────────────────────


def test_isomers_acetaminophen(client):
    r = client.post(
        "/isomers",
        json={
            "smiles": "CC(=O)NC1=CC=C(O)C=C1",
            "max_tautomers": 5,
            "max_stereoisomers": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_tautomers"] >= 1
    assert body["canonical_tautomer"] is not None
    assert body["n_tautomers"] <= 5


# ── Substructure search ─────────────────────────────────────────────────────


def test_substructure_with_explicit_candidates(client):
    r = client.post(
        "/substructure",
        json={
            "query": "c1ccccc1",
            "candidates": ["CCO", "c1ccccc1O", "c1ccncc1"],
            "limit": 10,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["n_matches"] == 1
    assert body["matches"][0]["smiles"] == "c1ccccc1O"


# ── Compare ─────────────────────────────────────────────────────────────────


def test_compare_two_similar_molecules(client):
    r = client.post(
        "/compare",
        json={
            "smiles_a": "CCO",
            "smiles_b": "CCN",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["a"]["canonical"] == "CCO"
    assert body["b"]["canonical"] == "CCN"
    assert 0.0 <= body["tanimoto_maccs"] <= 1.0
    assert len(body["descriptor_diff"]) > 0


# ── Standardize ─────────────────────────────────────────────────────────────


def test_standardize_simple(client):
    r = client.post("/standardize", json={"smiles": "CCO"})
    assert r.status_code == 200
    body = r.json()
    assert body["raw_canonical"] == "CCO"
    assert body["final_standardized"] == "CCO"


# ── MCS ─────────────────────────────────────────────────────────────────────


def test_mcs_two_aromatic_compounds(client):
    r = client.post(
        "/mcs",
        json={
            "smiles_a": "CC(=O)Oc1ccccc1C(=O)O",
            "smiles_b": "CC(=O)Nc1ccc(O)cc1",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_atoms"] > 0
    assert body["smarts"] is not None
    assert 0.0 <= body["a_coverage"] <= 1.0
    assert 0.0 <= body["b_coverage"] <= 1.0


# ── Structural alerts ──────────────────────────────────────────────────────


def test_alerts_clean_molecule(client):
    """Methanol is small + clean; it should not trip PAINS / Brenk / NIH."""
    r = client.post("/alerts", json={"smiles": "CO", "catalogs": ["PAINS", "BRENK"]})
    assert r.status_code == 200
    body = r.json()
    assert body["is_clean"] is True
    assert body["n_alerts"] == 0


def test_alerts_invalid_smiles(client):
    r = client.post("/alerts", json={"smiles": "XYZ"})
    assert r.status_code == 422


# ── Aggregated report ──────────────────────────────────────────────────────


def test_report_markdown(client):
    r = client.post(
        "/report",
        json={
            "smiles": "CCO",
            "include_admet": False,  # ADMET can be heavy; skip for fast CI
            "include_scaffold": True,
            "include_functional_groups": True,
            "include_descriptors": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "markdown" in body
    assert "# Molecule Report" in body["markdown"]
    assert "Scaffold" in body["markdown"]


# ── Library CRUD + CSV import/export ───────────────────────────────────────


class TestLibraryAPI:
    """End-to-end CRUD round-trip via the HTTP API."""

    def test_save_list_get_update_delete(self, client):
        # save
        r = client.post(
            "/library",
            json={
                "smiles": "CCO",
                "name": "Ethanol-API",
                "project": "_pytest",
                "tags": ["solvent"],
            },
        )
        assert r.status_code == 200
        compound = r.json()
        compound_id = compound["id"]

        # list
        r = client.get("/library", params={"project": "_pytest"})
        assert r.status_code == 200
        assert any(c["id"] == compound_id for c in r.json())

        # get
        r = client.get(f"/library/{compound_id}")
        assert r.status_code == 200
        assert r.json()["smiles"] == "CCO"

        # update
        r = client.patch(f"/library/{compound_id}", json={"name": "RenamedEthanol"})
        assert r.status_code == 200
        assert r.json()["name"] == "RenamedEthanol"

        # delete
        r = client.delete(f"/library/{compound_id}")
        assert r.status_code == 200
        # second delete = 404
        r = client.delete(f"/library/{compound_id}")
        assert r.status_code == 404

    def test_csv_export(self, client):
        r = client.get("/library/export/csv")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        text = r.text
        assert text.startswith("id,smiles,name,project,tags")

    def test_csv_import_bulk(self, client):
        r = client.post(
            "/library/import",
            json={
                "rows": [
                    {"smiles": "CCO", "name": "Ethanol-import", "project": "_import_test"},
                    {"smiles": "CCN", "name": "Ethylamine-import", "project": "_import_test"},
                ]
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_added"] == 2
        assert body["n_failed"] == 0

        # cleanup
        listed = client.get("/library", params={"project": "_import_test"}).json()
        for c in listed:
            client.delete(f"/library/{c['id']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
