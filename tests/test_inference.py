import pytest
from fastapi.testclient import TestClient

from molprop.serving.api import app, ml_models


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client_instance:
        yield client_instance


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "status" in data


def test_predict_no_model(client):
    """When no model is loaded, predict should return 503."""
    if ml_models.get("model") is None:
        response = client.post("/predict", json={"smiles": "CCO"})
        assert response.status_code == 503


def test_predict_invalid_smiles(client):
    """Test that invalid SMILES returns 400."""
    removed = False
    if not ml_models.get("model"):
        ml_models["model"] = "dummy"
        removed = True

    response = client.post("/predict", json={"smiles": "INVALID_SMILES123"})
    assert response.status_code == 400
    assert "Invalid SMILES string" in response.json()["detail"]

    if removed:
        ml_models["model"] = None


def test_predict_valid_smiles(client):
    response = client.post("/predict", json={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "explain": True})
    if not ml_models.get("model"):
        assert response.status_code == 503
    else:
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "task_1" in data["predictions"]


def test_batch_predict_no_model(client):
    """When no model is loaded, batch predict should return 503."""
    if ml_models.get("model") is None:
        response = client.post("/predict/batch", json={"smiles_list": ["CCO", "c1ccccc1"]})
        assert response.status_code == 503


def test_batch_predict_with_model(client):
    """Test batch prediction endpoint returns list of results."""
    if ml_models.get("model") is not None:
        response = client.post(
            "/predict/batch",
            json={"smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3


def test_response_has_timing_header(client):
    """Verify the timing middleware adds X-Process-Time."""
    response = client.get("/health")
    assert "x-process-time" in response.headers


# ── /descriptors ──────────────────────────────────────────────────────────────────────


def test_descriptors_valid_smiles(client):
    """Valid SMILES should return 18 descriptors."""
    response = client.post("/descriptors", json={"smiles": "c1ccccc1"})
    assert response.status_code == 200
    data = response.json()
    assert data["error"] is None
    assert len(data["descriptors"]) == 18
    assert "MolLogP" in data["descriptors"]
    assert "FractionCSP3" in data["descriptors"]


def test_descriptors_with_maccs(client):
    """include_fingerprint=True should add a 167-element list."""
    response = client.post(
        "/descriptors", json={"smiles": "c1ccccc1", "include_fingerprint": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["maccs_fingerprint"] is not None
    assert len(data["maccs_fingerprint"]) == 167


def test_descriptors_invalid_smiles(client):
    response = client.post("/descriptors", json={"smiles": "NOT_VALID"})
    assert response.status_code == 200
    assert response.json()["error"] is not None


# ── /lipinski ────────────────────────────────────────────────────────────────────────


def test_lipinski_aspirin(client):
    """Aspirin should pass Ro5 with no violations."""
    response = client.get("/lipinski", params={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"})
    assert response.status_code == 200
    data = response.json()
    assert data["passes"] is True
    assert data["violations"] == []
    assert data["MW"] is not None
    assert data["LogP"] is not None


def test_lipinski_invalid_smiles(client):
    response = client.get("/lipinski", params={"smiles": "NOT_A_MOLECULE"})
    assert response.status_code == 200
    assert response.json()["error"] is not None


# ── /conformer ────────────────────────────────────────────────────────────────────────


def test_conformer_valid_smiles(client):
    """Valid SMILES should return a non-empty PDB block."""
    response = client.post("/conformer", json={"smiles": "c1ccccc1"})
    assert response.status_code == 200
    data = response.json()
    assert data["error"] is None
    assert data["pdb_block"] is not None
    assert "ATOM" in data["pdb_block"] or "HETATM" in data["pdb_block"]
    assert data["num_atoms"] is not None and data["num_atoms"] > 0


def test_conformer_invalid_smiles(client):
    response = client.post("/conformer", json={"smiles": "NOT_VALID_XYZ"})
    assert response.status_code == 200
    assert response.json()["error"] is not None


# ── /generate/status ───────────────────────────────────────────────────────────────


def test_generate_status_endpoint(client):
    """Should return vae_loaded bool and optional latent_dim."""
    response = client.get("/generate/status")
    assert response.status_code == 200
    data = response.json()
    assert "vae_loaded" in data
    assert isinstance(data["vae_loaded"], bool)


# ── /search ────────────────────────────────────────────────────────────────────────────


def test_search_no_model(client):
    """Returns 503 when no model is loaded."""
    if ml_models.get("model") is None:
        response = client.get("/search", params={"smiles": "c1ccccc1"})
        assert response.status_code == 503


def test_search_invalid_topk(client):
    """top_k out of range should return 400."""
    if ml_models.get("model") is not None:
        response = client.get("/search", params={"smiles": "c1ccccc1", "top_k": 999})
        assert response.status_code == 400


def test_search_invalid_smiles(client):
    """Invalid SMILES should return 400."""
    if ml_models.get("model") is not None:
        response = client.get("/search", params={"smiles": "NOT_VALID"})
        assert response.status_code == 400


# ── /compare ───────────────────────────────────────────────────────────────────


def test_compare_valid_pair(client):
    """Two valid SMILES should return profiles and a Tanimoto score."""
    response = client.post(
        "/compare",
        json={"smiles_a": "c1ccccc1", "smiles_b": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "molecule_a" in data
    assert "molecule_b" in data
    assert "tanimoto_similarity" in data
    assert data["tanimoto_similarity"] is not None
    assert 0.0 <= data["tanimoto_similarity"] <= 1.0


def test_compare_identical_molecules(client):
    """Same molecule should yield Tanimoto similarity of 1.0."""
    response = client.post(
        "/compare",
        json={"smiles_a": "c1ccccc1", "smiles_b": "c1ccccc1"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tanimoto_similarity"] == 1.0


def test_compare_descriptors_present(client):
    """Profiles should include 18 physicochemical descriptors."""
    response = client.post(
        "/compare",
        json={"smiles_a": "c1ccccc1", "smiles_b": "CCO"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["molecule_a"]["descriptors"]) == 18
    assert len(data["molecule_b"]["descriptors"]) == 18


def test_compare_lipinski_present(client):
    """Each profile must expose Lipinski Ro5 results."""
    response = client.post(
        "/compare",
        json={"smiles_a": "c1ccccc1", "smiles_b": "CCO"},
    )
    assert response.status_code == 200
    data = response.json()
    for profile_key in ("molecule_a", "molecule_b"):
        ro5 = data[profile_key]["lipinski"]
        assert ro5 is not None
        assert "passes" in ro5
        assert "MW" in ro5


def test_compare_one_invalid_smiles(client):
    """One invalid SMILES should produce an error profile but not crash."""
    response = client.post(
        "/compare",
        json={"smiles_a": "NOT_VALID_XYZ", "smiles_b": "c1ccccc1"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["molecule_a"]["error"] is not None
    assert data["tanimoto_similarity"] is None
