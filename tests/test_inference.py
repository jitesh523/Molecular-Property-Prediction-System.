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
