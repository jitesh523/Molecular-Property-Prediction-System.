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

def test_predict_invalid_smiles(client):
    # Set a dummy if not loaded so we hit the standardization error
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
        assert response.status_code == 500
        assert "Model is not loaded" in response.json()["detail"]
    else:
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "task_1" in data["predictions"]
        if ml_models.get("explainer"):
            assert "explanation" in data
            assert "atom_importance" in data["explanation"]
