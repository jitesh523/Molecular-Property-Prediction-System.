"""
Analyze model errors to identify chemical biases.

Detects the largest outliers (molecules the model missed most significantly)
and calculates their physicochemical properties (MW, LogP, HBD, HBA) to
determine if the model fails on specific chemical classes.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from rich.console import Console
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
console = Console()


def calculate_properties(smiles: str) -> dict:
    """Calculate basic RDKit descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Model Errors")
    parser.add_argument(
        "--preds", type=str, required=True, help="Path to residuals/predictions CSV"
    )
    parser.add_argument("--dataset", type=str, default="bbbp", help="Dataset name")
    parser.add_argument("--top_n", type=int, default=20, help="Number of outliers to analyze")
    parser.add_argument("--output_dir", type=str, default="results/analysis/errors")
    parser.add_argument("--explain", action="store_true", help="Generate GNN explanations for outliers")
    parser.add_argument("--model_type", type=str, default="gat", help="GNN type for explanations")
    parser.add_argument("--weights", type=str, help="Path to weights for explanations")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Predictions/Residuals
    # Expects CSV with: std_smiles, y_true, y_pred
    if not Path(args.preds).exists():
        log.error(f"Prediction file not found: {args.preds}")
        return

    df = pd.read_csv(args.preds)
    if "error" not in df.columns:
        df["error"] = (df["y_true"] - df["y_pred"]).abs()

    log.info(f"Analyzing top {args.top_n} outliers for {args.dataset}...")

    # 2. Get Top Outliers
    outliers = df.sort_values(by="error", ascending=False).head(args.top_n)

    # 3. Calculate Properties for Outliers and Background
    # For subset of background to compare
    bg_df = df.sample(min(500, len(df)), random_state=42)

    def get_df_props(target_df):
        props = []
        for s in target_df["std_smiles"]:
            p = calculate_properties(s)
            if p:
                props.append(p)
        return pd.DataFrame(props)

    outlier_props = get_df_props(outliers)
    bg_props = get_df_props(bg_df)

    outlier_props["group"] = "Outliers"
    bg_props["group"] = "Background"
    combined = pd.concat([outlier_props, bg_props])

    # 4. Display Summary Table
    table = Table(title=f"Top {args.top_n} Outliers: Chemical Properties")
    table.add_column("Property", style="cyan")
    table.add_column("Outliers (Mean)", justify="right")
    table.add_column("Background (Mean)", justify="right")

    for col in ["MW", "LogP", "HBD", "HBA", "TPSA"]:
        table.add_row(col, f"{outlier_props[col].mean():.2f}", f"{bg_props[col].mean():.2f}")
    console.print(table)

    # 5. Visual Correlation Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_style("whitegrid")

    sns.boxplot(data=combined, x="group", y="MW", ax=axes[0], palette="muted")
    axes[0].set_title("Molecular Weight Distribution")

    sns.boxplot(data=combined, x="group", y="LogP", ax=axes[1], palette="muted")
    axes[1].set_title("LogP Distribution")

    plt.tight_layout()
    plot_path = out_dir / f"{args.dataset}_error_dist.png"
    plt.savefig(plot_path, dpi=300)
    log.info(f"Saved property distribution plot to {plot_path}")

    # 6. Save Outlier Report
    report_path = out_dir / f"{args.dataset}_outliers.csv"
    outliers.to_csv(report_path, index=False)
    log.info(f"Saved outlier report to {report_path}")

    # 7. (Optional) Explain Outliers
    if args.explain:
        from molprop.serving.load_model import load_gnn_model
        from molprop.models.explain import get_explainer, explain_graph
        from molprop.models.visualize_explanations import get_explanation_image
        from molprop.features.graphs import smiles_to_graph
        import torch

        if not args.weights:
            log.error("Weights path required for explanations.")
            return

        log.info("Generating explanations for top outliers...")
        model = load_gnn_model(args.model_type, args.weights)
        explainer = get_explainer(
            model, 
            task_type="regression" if "regression" in args.preds else "binary_classification",
            algorithm="captum"
        )
        
        exp_dir = out_dir / "explanations"
        exp_dir.mkdir(exist_ok=True)

        for i, row in outliers.iterrows():
            s = row["std_smiles"]
            data = smiles_to_graph(s)
            if data:
                explanation = explain_graph(explainer, data.x, data.edge_index, data.edge_attr)
                svg = get_explanation_image(s, explanation)
                svg_path = exp_dir / f"outlier_{i}.svg"
                with open(svg_path, "w") as f:
                    f.write(svg)
        log.info(f"Saved {len(outliers)} outlier explanations to {exp_dir}")


if __name__ == "__main__":
    main()
