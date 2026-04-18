"""
Automated benchmark table generator.

Runs all available models (baselines + GNNs) across all processed datasets
and produces a unified markdown results table and CSV.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# torch import moved to run_gnn_benchmark to avoid conflicts with XGBoost/OpenMP
from rich.console import Console
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

from molprop.data.splits import random_scaffold_split
from molprop.features.fingerprints import batch_smiles_to_morgan
from molprop.features.graphs import smiles_to_graph
from molprop.models.baselines import BaselineModel
from molprop.models.evaluate import (
    plot_calibration_curve,
    plot_error_distribution,
    plot_pr_curve,
    plot_regression_scatter,
    plot_roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
console = Console()

# Dataset registry: name -> (task_type, target_col_override_or_None)
DATASET_REGISTRY = {
    "delaney": ("regression", None),
    "freesolv": ("regression", None),
    "lipo": ("regression", None),
    "bbbp": ("classification", None),
}


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R²": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(y_true, y_score, y_pred) -> Dict[str, float]:
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


def run_baseline_benchmark(
    dataset_name: str, task_type: str, processed_dir: Path, generate_plots: bool = False
) -> List[Dict]:
    """Run RF and XGBoost baselines on a dataset."""
    data_path = processed_dir / dataset_name / "processed.csv"
    if not data_path.exists():
        log.warning(f"Skipping {dataset_name}: no processed data at {data_path}")
        return []

    df = pd.read_csv(data_path)
    smiles_list = df["std_smiles"].tolist()
    target_cols = [c for c in df.columns if c != "std_smiles"]
    if not target_cols:
        return []
    target = target_cols[0]

    log.info(f"  Featurizing {len(smiles_list)} SMILES...")
    x = batch_smiles_to_morgan(smiles_list)
    y = df[target].values
    log.info("  Standardizing targets...")

    valid_mask = ~pd.isna(y)
    x, y = x[valid_mask], y[valid_mask]
    smiles_valid = [smiles_list[i] for i in range(len(smiles_list)) if valid_mask[i]]
    log.info(f"  Splitting {len(smiles_valid)} valid samples...")
    train_idx, val_idx, test_idx = random_scaffold_split(smiles_valid)
    log.info("  Split complete. Training models...")
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    results = []
    for name, mtype in [("RF", "rf"), ("XGBoost", "xgb")]:
        log.info(f"    Training {name}...")
        model = BaselineModel(
            model_type=mtype,
            task_type=task_type,
            params={"n_estimators": 100, "random_state": 42, "n_jobs": 1},
        )
        model.train(x_train, y_train)
        log.info(f"    Evaluating {name}...")

        if task_type == "regression":
            y_pred = model.predict(x_test)
            metrics = evaluate_regression(y_test, y_pred)
            if generate_plots:
                ROOT = Path(__file__).resolve().parent.parent
                plots_dir = ROOT / "results" / "plots"
                plot_regression_scatter(y_test, y_pred, name, dataset_name, plots_dir)
                plot_error_distribution(y_test, y_pred, name, dataset_name, plots_dir)
        else:
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)
            y_score = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics = evaluate_classification(y_test, y_score, y_pred)
            if generate_plots:
                ROOT = Path(__file__).resolve().parent.parent
                plots_dir = ROOT / "results" / "plots"
                plot_roc_curve(y_test, y_score, name, dataset_name, plots_dir)
                plot_pr_curve(y_test, y_score, name, dataset_name, plots_dir)
                plot_calibration_curve(y_test, y_score, name, dataset_name, plots_dir)

        results.append({"Dataset": dataset_name, "Model": name, "Split": "Scaffold", **metrics})

    return results


def run_gnn_benchmark(
    dataset_name: str, task_type: str, processed_dir: Path, root_dir: Path
) -> List[Dict]:
    """Run GNN models on a dataset (only if trained weights exist)."""
    import torch

    with torch.no_grad():
        data_path = processed_dir / dataset_name / "processed.csv"
        if not data_path.exists():
            return []

        df = pd.read_csv(data_path)
        target_cols = [c for c in df.columns if c != "std_smiles"]
        if not target_cols:
            return []
        target = target_cols[0]

        # Build graphs
        dataset = []
        for _, row in df.iterrows():
            g = smiles_to_graph(row["std_smiles"], y=row[target])
            if g:
                dataset.append(g)

        if not dataset:
            return []

        smiles_list = [g.smiles for g in dataset]
        _, _, test_idx = random_scaffold_split(smiles_list)

        if not test_idx:
            return []

        # Device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        test_data = [dataset[i] for i in test_idx]
        test_loader = DataLoader(test_data, batch_size=64)
        in_dim = dataset[0].num_node_features

        results = []
        from molprop.models.gnn_gat import GATModel
        from molprop.models.gnn_gcn import GCNModel
        from molprop.models.gnn_mpnn import MPNNModel

        model_configs = [
            (
                "GCN",
                GCNModel,
                {
                    "in_dim": in_dim,
                    "hidden_dim": 128,
                    "out_dim": 1,
                    "num_layers": 3,
                    "dropout": 0.2,
                },
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

        for name, model_cls, kwargs in model_configs:
            weights_path = root_dir / f"best_model_{name.lower()}_{dataset_name}.pt"
            if not weights_path.exists():
                log.info(f"  Skipping {name} on {dataset_name}: no weights at {weights_path}")
                continue

            model = model_cls(**kwargs).to(device)
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            model.eval()

            y_true_all, y_pred_all = [], []
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                y_true_all.append(data.y.view(-1, 1).cpu().numpy())
                if task_type == "classification":
                    y_pred_all.append(torch.sigmoid(out).cpu().numpy())
                else:
                    y_pred_all.append(out.cpu().numpy())

            y_true = np.concatenate(y_true_all)
            y_pred = np.concatenate(y_pred_all)

            if task_type == "regression":
                metrics = evaluate_regression(y_true, y_pred)
            else:
                y_pred_binary = (y_pred > 0.5).astype(int)
                metrics = evaluate_classification(y_true, y_pred, y_pred_binary)

            results.append({"Dataset": dataset_name, "Model": name, "Split": "Scaffold", **metrics})

    return results

    return results


def generate_markdown_table(results_df: pd.DataFrame, task_type: str) -> str:
    """Convert results DataFrame to a markdown table string."""
    lines = []

    if task_type == "regression":
        cols = ["Dataset", "Model", "RMSE", "MAE", "R²"]
    else:
        cols = ["Dataset", "Model", "ROC-AUC", "PR-AUC", "MCC"]

    available_cols = [c for c in cols if c in results_df.columns]
    sub = results_df[available_cols]

    # Header
    lines.append("| " + " | ".join(available_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(available_cols)) + "|")

    for _, row in sub.iterrows():
        vals = []
        for c in available_cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main():
    # Fix for potential OpenMP runtime conflicts (common cause of segfaults with XGBoost/SKLearn)
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser(description="Generate benchmark results table")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to benchmark (default: all in registry)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model types to run: rf, xgb, gcn, gat, mpnn (default: all)",
    )
    parser.add_argument(
        "--skip-gnn",
        action="store_true",
        help="Skip GNN benchmarks (useful if no trained weights exist)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate evaluation plots (PR, ROC, scatter, etc.)",
    )
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    processed_dir = ROOT / "data" / "processed"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or list(DATASET_REGISTRY.keys())
    all_results = []

    for ds_name in datasets:
        if ds_name not in DATASET_REGISTRY:
            log.warning(f"Unknown dataset: {ds_name}, skipping.")
            continue

        task_type, _ = DATASET_REGISTRY[ds_name]
        log.info(f"\n{'=' * 60}")
        log.info(f"Benchmarking: {ds_name} ({task_type})")
        log.info(f"{'=' * 60}")

        # Baselines
        baseline_results = run_baseline_benchmark(
            ds_name, task_type, processed_dir, generate_plots=args.plots
        )
        all_results.extend(baseline_results)

        # GNNs
        if not args.skip_gnn:
            gnn_results = run_gnn_benchmark(ds_name, task_type, processed_dir, ROOT)
            all_results.extend(gnn_results)

    if not all_results:
        log.error("No results generated. Check data availability.")
        return

    results_df = pd.DataFrame(all_results)

    # Save CSV
    csv_path = results_dir / "benchmark_table.csv"
    results_df.to_csv(csv_path, index=False)
    log.info(f"\nCSV saved to {csv_path}")

    # Generate and save markdown tables (grouped by task type)
    md_lines = ["# Benchmark Results\n"]
    md_lines.append("All models evaluated using **scaffold split** (80/10/10).\n")

    for task_type in ["regression", "classification"]:
        subset = results_df[
            results_df["Dataset"].isin(
                [k for k, v in DATASET_REGISTRY.items() if v[0] == task_type]
            )
        ]
        if subset.empty:
            continue
        md_lines.append(f"\n## {task_type.capitalize()} Tasks\n")
        md_lines.append(generate_markdown_table(subset, task_type))
        md_lines.append("")

    md_path = results_dir / "benchmark_table.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log.info(f"Markdown table saved to {md_path}")

    # Pretty print
    console.print(f"\n[bold green]Benchmark complete! Results in {results_dir}[/bold green]")


if __name__ == "__main__":
    main()
