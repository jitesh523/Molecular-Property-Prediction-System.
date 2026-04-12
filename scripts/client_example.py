#!/usr/bin/env python3
"""
Client example for the Molecular Property Prediction API.

Demonstrates single prediction, batch prediction, and explanation
requests. Uses only the `requests` library — no project imports needed.

Usage:
    # Start the API first:
    #   uvicorn molprop.serving.api:app --reload
    #
    # Then run this script:
    python scripts/client_example.py
"""

import sys

import requests

BASE_URL = "http://localhost:8000"


def check_health():
    """Check if the API is running."""
    print("━" * 60)
    print("1. Health Check")
    print("━" * 60)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"   Status: {r.status_code}")
        print(f"   Response: {r.json()}")
        return r.json().get("model_loaded", False)
    except requests.ConnectionError:
        print("   ✗ API is not running. Start with:")
        print("     uvicorn molprop.serving.api:app --reload")
        return False


def get_model_info():
    """Get metadata about the loaded model."""
    print("\n" + "━" * 60)
    print("2. Model Info")
    print("━" * 60)
    r = requests.get(f"{BASE_URL}/model/info", timeout=5)
    info = r.json()
    for k, v in info.items():
        print(f"   {k}: {v}")


def single_prediction():
    """Make a single prediction with explanation."""
    print("\n" + "━" * 60)
    print("3. Single Prediction (Aspirin)")
    print("━" * 60)

    payload = {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "explain": True,
    }
    print(f"   Input SMILES: {payload['smiles']}")

    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    result = r.json()

    print(f"   Standardized: {result.get('standardized_smiles', 'N/A')}")
    preds = result.get("predictions", {})
    for task, val in preds.items():
        print(f"   Prediction ({task}): {val:.4f}")

    explanation = result.get("explanation")
    if explanation:
        atoms = explanation.get("atom_importance", [])
        print(f"   Atom importance ({len(atoms)} atoms): {[round(a, 3) for a in atoms[:5]]}...")


def batch_prediction():
    """Make batch predictions for multiple molecules."""
    print("\n" + "━" * 60)
    print("4. Batch Prediction")
    print("━" * 60)

    drug_molecules = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "Penicillin V": "CC1(C)SC2C(NC(=O)COC3=CC=CC=C3)C(=O)N2C1C(=O)O",
    }

    payload = {"smiles_list": list(drug_molecules.values()), "explain": False}

    r = requests.post(f"{BASE_URL}/predict/batch", json=payload, timeout=30)
    results = r.json()

    print(f"   Submitted {len(payload['smiles_list'])} molecules")
    print(f"   Process time: {r.headers.get('X-Process-Time', 'N/A')}")
    print()

    names = list(drug_molecules.keys())
    for name, res in zip(names, results, strict=False):
        pred = res.get("predictions", {}).get("task_1", "N/A")
        error = res.get("error")
        if error:
            print(f"   {name:15s} → Error: {error}")
        else:
            print(f"   {name:15s} → {pred:.4f}")


def equivalent_curl():
    """Print equivalent curl commands for reference."""
    print("\n" + "━" * 60)
    print("5. Equivalent curl Commands")
    print("━" * 60)
    print(
        """
   # Single prediction:
   curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "explain": true}'

   # Batch prediction:
   curl -X POST http://localhost:8000/predict/batch \\
     -H "Content-Type: application/json" \\
     -d '{"smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"]}'

   # Model info:
   curl http://localhost:8000/model/info

   # Interactive docs:
   open http://localhost:8000/docs
"""
    )


def main():
    model_loaded = check_health()

    if not model_loaded:
        print("\n   ⚠ Model not loaded — predictions will return 503.")
        print("   Set MODEL_WEIGHTS env var to a valid .pt file path.\n")
        sys.exit(1)

    get_model_info()
    single_prediction()
    batch_prediction()
    equivalent_curl()

    print("━" * 60)
    print("✓ All examples completed successfully!")
    print("━" * 60)


if __name__ == "__main__":
    main()
