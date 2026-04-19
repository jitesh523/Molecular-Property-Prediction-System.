import logging
import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from hydra import initialize, compose

from molprop.features.graphs import smiles_to_graph
from molprop.serving.load_model import load_gnn_model
from molprop.serving.vector_db import vector_store

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def index_dataset(dataset_name: str, model_type: str, weights_path: str):
    """
    Encodes a dataset and populates the in-memory Qdrant instance.
    """
    root_dir = Path(__file__).resolve().parent.parent
    data_path = root_dir / "data" / "processed" / dataset_name / "processed.csv"
    
    if not data_path.exists():
        log.error(f"Processed data not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Load model
    device = torch.device("cpu") # Use CPU for indexing simplicity
    model = load_gnn_model(model_type, weights_path, device=device)
    model.eval()

    log.info(f"Indexing {len(df)} molecules from {dataset_name}...")
    
    points = []
    vector_size = None
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row["std_smiles"]
        target = row.get("target", row.get(df.columns[-1])) # Guessing target if not explicit
        
        g = smiles_to_graph(smiles)
        if g:
            embedding = model.encode(g).squeeze(0).cpu().numpy().tolist()
            if vector_size is None:
                vector_size = len(embedding)
                vector_store.create_collection(vector_size)
            
            points.append({
                "id": i,
                "vector": embedding,
                "payload": {
                    "smiles": smiles,
                    "dataset": dataset_name,
                    "task_value": float(target) if pd.notnull(target) else 0.0
                }
            })
            
            # Batch upsert every 100 points
            if len(points) >= 100:
                vector_store.upsert_molecules(points)
                points = []

    # Final upsert
    if points:
        vector_store.upsert_molecules(points)
    
    log.info("Indexing complete.")

def test_search(query_smiles: str, model_type: str, weights_path: str):
    log.info(f"Testing search for: {query_smiles}")
    device = torch.device("cpu")
    model = load_gnn_model(model_type, weights_path, device=device)
    
    g = smiles_to_graph(query_smiles)
    if g:
        query_vec = model.encode(g).squeeze(0).cpu().numpy().tolist()
        results = vector_store.search_similar(query_vec, top_k=5)
        
        log.info("Top 5 Similar Molecules:")
        for r in results:
            log.info(f" - {r['smiles']} (Score: {r['score']:.4f})")

if __name__ == "__main__":
    # Example usage
    # python scripts/index_molecules.py bbbp gcn best_model_gcn_bbbp.pt
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bbbp"
    m_type = sys.argv[2] if len(sys.argv) > 2 else "gcn"
    weights = sys.argv[3] if len(sys.argv) > 3 else f"best_model_{m_type}_{dataset}.pt"
    
    index_dataset(dataset, m_type, weights)
    
    # Test with a known molecule (Caffeine)
    test_search("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", m_type, weights)
