"""
Ablation study: Fingerprint vs Descriptors vs Graph vs Hybrid.

Runs all model × representation combinations on a given dataset
using scaffold split, producing a unified comparison table and
grouped bar chart.

This is the Week 6 roadmap deliverable — structured evidence of
when GNNs help versus when fingerprints suffice.

Usage:
    python scripts/run_ablation.py --dataset delaney --task regression
    python scripts/run_ablation.py --dataset bbbp --task classification
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from molprop.data.splits import random_scaffold_split
from molprop.features.descriptors import batch_smiles_to_descriptors
from molprop.features.fingerprints import batch_smiles_to_morgan
from molprop.models.baselines import BaselineModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
console = Console()


# ── Metrics ────────────────────────────────────────────────────────────────────


def eval_regression(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R²": float(r2_score(y_true, y_pred)),
    }


def eval_classification(y_true, y_score, y_pred):
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = 0.5
    try:
        pr = float(average_precision_score(y_true, y_score))
    except ValueError:
        pr = 0.0
    return {
        "ROC-AUC": auc,
        "PR-AUC": pr,
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


# ── Feature builders ──────────────────────────────────────────────────────────


def build_features(smiles_list, representation):
    """Build feature matrix for the given representation type."""
    if representation == "fingerprint":
        return batch_smiles_to_morgan(smiles_list)
    elif representation == "descriptors":
        return batch_smiles_to_descriptors(smiles_list)
    elif representation == "hybrid":
        fps = batch_smiles_to_morgan(smiles_list)
        descs = batch_smiles_to_descriptors(smiles_list)
        return np.concatenate([fps, descs], axis=1)
    else:
        raise ValueError(f"Unknown representation: {representation}")


# ── GNN evaluation (uses pre-trained weights) ─────────────────────────────────


def eval_gnn_if_available(dataset_name, task_type, smiles_list, y, test_idx, root_dir):
    """Attempt to load and evaluate pre-trained GNN weights."""
    import torch
    from torch_geometric.loader import DataLoader

    from molprop.features.graphs import smiles_to_graph
    from molprop.models.gnn_gat import GATModel
    from molprop.models.gnn_gcn import GCNModel
    from molprop.models.gnn_mpnn import MPNNModel

    # Build graph dataset
    dataset = []
    idx_map = {}  # original_idx -> graph_idx
    for orig_idx, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi, y=y[orig_idx])
        if g is not None:
            idx_map[orig_idx] = len(dataset)
            dataset.append(g)

    if not dataset:
        return []

    in_dim = dataset[0].num_node_features
    device = torch.device("cpu")

    gnn_configs = [
        (
            "GCN",
            GCNModel,
            {"in_dim": in_dim, "hidden_dim": 128, "out_dim": 1, "num_layers": 3, "dropout": 0.2},
        ),
        (
            "GAT",
            GATModel,
            {
                "in_dim": in_dim,
                "hidden_dim": 128,
                "out_dim": 1,
                "num_layers": 3,
                "dropout": 0.2,
                "heads": 4,
            },
        ),
        (
            "MPNN",
            MPNNModel,
            {
                "in_dim": in_dim,
                "hidden_dim": 128,
                "out_dim": 1,
                "num_layers": 3,
                "dropout": 0.2,
                "edge_dim": 4,
            },
        ),
    ]

    results = []
    for name, cls, kwargs in gnn_configs:
        weights_path = root_dir / f"best_model_{name.lower()}_{dataset_name}.pt"
        if not weights_path.exists():
            # Also check variant naming
            weights_path = root_dir / f"best_{name.lower()}_{dataset_name}.pt"
        if not weights_path.exists():
            log.info(f"  ⊘ {name}: no weights found, skipping.")
            continue

        model = cls(**kwargs).to(device)
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except RuntimeError as e:
            log.warning(f"  ⊘ {name}: weight loading failed ({e}), skipping.")
            continue
        model.eval()

        # Get test graphs
        test_graphs = [dataset[idx_map[i]] for i in test_idx if i in idx_map]
        if not test_graphs:
            continue

        loader = DataLoader(test_graphs, batch_size=64)
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data)
                y_true_all.append(data.y.view(-1).cpu().numpy())
                if task_type == "classification":
                    y_pred_all.append(torch.sigmoid(out).view(-1).cpu().numpy())
                else:
                    y_pred_all.append(out.view(-1).cpu().numpy())

        yt = np.concatenate(y_true_all)
        yp = np.concatenate(y_pred_all)

        if task_type == "regression":
            metrics = eval_regression(yt, yp)
        else:
            metrics = eval_classification(yt, yp, (yp > 0.5).astype(int))

        results.append({"Representation": "graph", "Model": name, **metrics})
        log.info(f"  ✓ {name} (graph): {metrics}")

    return results


# ── Main ablation runner ──────────────────────────────────────────────────────


def run_ablation(dataset_name, task_type, root_dir):
    """Run the full ablation study for a dataset."""
    processed_dir = root_dir / "data" / "processed"
    data_path = processed_dir / dataset_name / "processed.csv"

    if not data_path.exists():
        log.error(f"Dataset not found: {data_path}")
        return None

    df = pd.read_csv(data_path)
    smiles_list = df["std_smiles"].tolist()
    target_cols = [c for c in df.columns if c != "std_smiles"]
    if not target_cols:
        return None
    target = target_cols[0]
    y = df[target].values

    # Drop NaN
    valid = ~pd.isna(y)
    smiles_valid = [smiles_list[i] for i in range(len(smiles_list)) if valid[i]]
    y_valid = y[valid]

    # Scaffold split
    train_idx, val_idx, test_idx = random_scaffold_split(smiles_valid)

    all_results = []

    # ── Tabular representations ───────────────────────────────────────────
    representations = [
        ("fingerprint", "Morgan FP (2048-bit)"),
        ("descriptors", "RDKit Descriptors (9)"),
        ("hybrid", "FP + Descriptors"),
    ]
    model_types = [("RF", "rf"), ("XGBoost", "xgb")]

    for rep_key, rep_label in representations:
        log.info(f"\n── {rep_label} ──")
        try:
            x = build_features(smiles_valid, rep_key)
        except Exception as e:
            log.warning(f"  Feature extraction failed for {rep_key}: {e}")
            continue

        x_train, y_train = x[train_idx], y_valid[train_idx]
        x_test, y_test = x[test_idx], y_valid[test_idx]

        for model_name, model_type in model_types:
            model = BaselineModel(
                model_type=model_type,
                task_type=task_type,
                params={"n_estimators": 500, "random_state": 42},
            )
            model.train(x_train, y_train)

            if task_type == "regression":
                y_pred = model.predict(x_test)
                metrics = eval_regression(y_test, y_pred)
            else:
                y_pred = model.predict(x_test)
                y_proba = model.predict_proba(x_test)
                y_score = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics = eval_classification(y_test, y_score, y_pred)

            all_results.append({"Representation": rep_label, "Model": model_name, **metrics})
            log.info(f"  ✓ {model_name}: {metrics}")

    # ── Graph representations (GNNs) ──────────────────────────────────────
    log.info("\n── Graph Neural Networks ──")
    gnn_results = eval_gnn_if_available(
        dataset_name, task_type, smiles_valid, y_valid, test_idx, root_dir
    )
    all_results.extend(gnn_results)

    return all_results


def generate_ablation_chart(results_df, task_type, dataset_name, output_dir):
    """Generate a grouped bar chart comparing all representation × model combos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metric = "RMSE" if task_type == "regression" else "ROC-AUC"
    if metric not in results_df.columns:
        return

    # Pivot for grouped bar chart
    pivot = results_df.pivot_table(
        values=metric, index="Model", columns="Representation", aggfunc="first"
    )

    # Colors
    colors = ["#4361ee", "#7209b7", "#f72585", "#06d6a0", "#ff6b35", "#1b9aaa"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot.index))
    width = 0.8 / len(pivot.columns)

    for i, col in enumerate(pivot.columns):
        vals = pivot[col].values
        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=col,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels on bars
        for bar, val in zip(bars, vals, strict=False):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(
        f"Ablation Study — {metric} by Representation × Model ({dataset_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.legend(title="Representation", fontsize=9, title_fontsize=10, loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    path = output_dir / f"ablation_chart_{dataset_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Chart saved to {path}")


def generate_markdown_table(results_df, task_type):
    """Generate a markdown table from ablation results."""
    if task_type == "regression":
        cols = ["Representation", "Model", "RMSE", "MAE", "R²"]
    else:
        cols = ["Representation", "Model", "ROC-AUC", "PR-AUC", "MCC"]

    available = [c for c in cols if c in results_df.columns]
    lines = []
    lines.append("| " + " | ".join(available) + " |")
    lines.append("|" + "|".join(["---"] * len(available)) + "|")
    for _, row in results_df[available].iterrows():
        vals = []
        for c in available:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--dataset",
        type=str,
        default="delaney",
        help="Dataset name (default: delaney)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "results" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold cyan]═══ Ablation Study: {args.dataset} ({args.task}) ═══[/bold cyan]\n"
    )

    results = run_ablation(args.dataset, args.task, root_dir)

    if not results:
        log.error("No results generated.")
        return

    results_df = pd.DataFrame(results)

    # Save CSV
    csv_path = output_dir / f"ablation_{args.dataset}.csv"
    results_df.to_csv(csv_path, index=False)
    log.info(f"\nCSV saved to {csv_path}")

    # Save markdown
    md_content = f"# Ablation Study: {args.dataset}\n\n"
    md_content += f"Task: **{args.task}** | Split: **Scaffold (80/10/10)**\n\n"
    md_content += generate_markdown_table(results_df, args.task)
    md_path = output_dir / f"ablation_{args.dataset}.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    log.info(f"Markdown saved to {md_path}")

    # Generate chart
    generate_ablation_chart(results_df, args.task, args.dataset, output_dir)

    # Rich console output
    table = Table(title=f"Ablation: {args.dataset}")
    for col in results_df.columns:
        table.add_column(col)
    for _, row in results_df.iterrows():
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)

    console.print(f"\n[bold green]Ablation complete! Results in {output_dir}[/bold green]")


if __name__ == "__main__":
    main()
