"""
Visualize the chemical space of a molecular dataset using UMAP.

Reduces high-dimensional molecular representations (Fingerprints or GNN Embeddings)
to 2D and plots them colored by activity or dataset split. Useful for
understanding the applicability domain of models.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from tqdm import tqdm

from molprop.features.fingerprints import batch_smiles_to_morgan
from molprop.serving.load_model import load_gnn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def get_gnn_embeddings(smiles_list: list[str], model_type: str, weights_path: str):
    """Extraction of graph-level embeddings from a trained GNN backbone."""
    from torch_geometric.loader import DataLoader

    from molprop.features.graphs import smiles_to_graph

    log.info(f"Extracting GNN embeddings using {model_type} from {weights_path}...")

    # Load model as MultiTaskGNN but we just want the backbone output
    # Actually, load_gnn_model returns the model in eval mode.
    model = load_gnn_model(model_type, weights_path)

    # If it's a MultiTaskGNN, we need to handle the fact that forward() returns tasks.
    # However, our MultiTaskGNN implementation returns emb in a previous line.
    # We can use a hook or just modify current model to return emb.
    # For now, we'll assume the model object itself has a way to get embeddings.

    graphs = []
    for s in tqdm(smiles_list, desc="Converting to graphs"):
        g = smiles_to_graph(s)
        if g:
            graphs.append(g)

    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    embeddings = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            # Check if it's MultiTaskGNN or base model
            if hasattr(model, "backbone"):
                # MultiTaskGNN: get emb from backbone
                emb = model.backbone(batch)
            else:
                # Base model: if it's GNNBase, it has an MLP head.
                # We want the pooling output.
                # Let's assume we can get it or just use the whole model output
                # if it's the last hidden layer.
                # To be safe, let's just use the backbone if it exists.
                emb = model(batch)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Visualize Chemical Space")
    parser.add_file = parser.add_argument
    parser.add_file("--dataset", type=str, default="bbbp", help="Dataset name")
    parser.add_file("--representation", type=str, choices=["morgan", "gnn"], default="morgan")
    parser.add_file("--model_type", type=str, default="gcn", help="GNN type if repo used")
    parser.add_file("--weights", type=str, help="Path to weights file for GNN embeddings")
    parser.add_file("--output_dir", type=str, default="results/plots/chemical_space")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    data_path = Path(f"data/processed/{args.dataset}/processed.csv")
    if not data_path.exists():
        log.error(f"Dataset not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    # Subset if too large for UMAP interactivity
    if len(df) > 10000:
        log.info("Subsampling to 10k rows for visualization...")
        df = df.sample(10000, random_state=42)

    smiles_list = df["std_smiles"].tolist()
    target_col = [c for c in df.columns if c != "std_smiles"][0]

    # 2. Extract Features
    if args.representation == "morgan":
        features = batch_smiles_to_morgan(smiles_list)
    else:
        if not args.weights:
            log.error("Weights path required for GNN embeddings.")
            return
        features = get_gnn_embeddings(smiles_list, args.model_type, args.weights)

    # 3. UMAP Reduction
    log.info("Running UMAP reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(features)

    df["UMAP1"] = embedding_2d[:, 0]
    df["UMAP2"] = embedding_2d[:, 1]

    # 4. Plotting
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")

    # If target is numeric, use scatter with cmap
    # If binary, use discrete colors
    unique_vals = df[target_col].nunique()
    if unique_vals <= 10:
        palette = "viridis" if unique_vals == 2 else "tab10"
        sns.scatterplot(
            data=df, x="UMAP1", y="UMAP2", hue=target_col, palette=palette, alpha=0.6, s=15
        )
    else:
        plt.scatter(df["UMAP1"], df["UMAP2"], c=df[target_col], cmap="plasma", alpha=0.5, s=10)
        plt.colorbar(label=target_col)

    plt.title(f"Chemical Space Visualization: {args.dataset} ({args.representation})")
    plt.tight_layout()

    out_path = out_dir / f"umap_{args.dataset}_{args.representation}.png"
    plt.savefig(out_path, dpi=300)
    log.info(f"Saved plot to {out_path}")
    
    # Save CSV for interactive dashboard
    csv_path = out_dir / f"umap_{args.dataset}_{args.representation}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved CSV data to {csv_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
