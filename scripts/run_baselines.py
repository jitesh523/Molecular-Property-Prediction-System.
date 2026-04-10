import argparse
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from molprop.data.splits import random_scaffold_split, scaffold_kfold, scaffold_split
from molprop.features.fingerprints import batch_smiles_to_morgan
from molprop.models.baselines import BaselineModel
from molprop.models.explain_baselines import (
    get_shap_explanation,
    save_shap_report,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
console = Console()


def run_benchmark(
    dataset_name: str,
    task_type: str,
    processed_dir: Path,
    split_type: str = "random_scaffold",
    explain: bool = False,
):
    """
    Run baseline models on a specific dataset.
    """
    data_path = processed_dir / dataset_name / "processed.csv"
    if not data_path.exists():
        log.error(f"Processed data not found for {dataset_name} at {data_path}")
        return

    log.info(f"\n[bold blue]Running Benchmark: {dataset_name}[/bold blue]")
    df = pd.read_csv(data_path)

    # In processed.csv, std_smiles is the first column.
    # Remaining columns are targets.
    smiles_list = df["std_smiles"].tolist()
    target_cols = [c for c in df.columns if c != "std_smiles"]

    if not target_cols:
        log.error(f"No target columns found in {dataset_name}")
        return

    # For now, let's assume single-task for baselines or pick the first target
    target = target_cols[0]
    log.info(f"Target task: {target} ({task_type})")

    # 1. Featurization
    log.info("Generating Morgan Fingerprints...")
    x = batch_smiles_to_morgan(smiles_list)
    y = df[target].values

    # Handle multi-task missing values (if any, though processed.csv should be clean)
    valid_mask = ~pd.isna(y)
    x = x[valid_mask]
    y = y[valid_mask]
    smiles_list_valid = [smiles_list[i] for i in range(len(smiles_list)) if valid_mask[i]]

    # 2. Splitting
    log.info(f"Performing {split_type} Split...")
    if split_type == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(smiles_list_valid)
    else:
        train_idx, val_idx, test_idx = random_scaffold_split(smiles_list_valid)

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    log.info(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    if task_type == "classification":
        from collections import Counter

        log.info(f"Train distribution: {Counter(y_train)}")
        log.info(f"Val distribution: {Counter(y_val)}")
        log.info(f"Test distribution: {Counter(y_test)}")

    # 3. Training & Evaluation
    results = []

    models = [
        ("RF", "rf", {"n_estimators": 100, "random_state": 42}),
        ("XGBoost", "xgb", {"n_estimators": 100, "random_state": 42}),
    ]

    for name, mtype, params in models:
        log.info(f"Training {name}...")
        model = BaselineModel(model_type=mtype, task_type=task_type, params=params)
        model.train(x_train, y_train)

        # Eval on Val and Test
        val_metrics = model.evaluate(x_val, y_val)
        test_metrics = model.evaluate(x_test, y_test)

        results.append({"Model": name, "Split": "Validation", **val_metrics})
        results.append({"Model": name, "Split": "Test", **test_metrics})

        # SHAP Explanation
        if explain:
            log.info(f"Generating SHAP explanation for {name}...")
            ROOT = Path(__file__).resolve().parent.parent
            explanation_dir = ROOT / "results" / "explanations"
            shap_values = get_shap_explanation(model.model, x_test)
            save_shap_report(
                shap_values,
                smiles_list_valid,
                explanation_dir,
                dataset_name,
                name.lower(),
            )

    # 4. Display Results
    table = Table(title=f"Baseline Results: {dataset_name}")
    table.add_column("Model", style="cyan")
    table.add_column("Split", style="magenta")

    if task_type == "regression":
        table.add_column("RMSE", justify="right")
        table.add_column("MAE", justify="right")
        table.add_column("R2", justify="right")
        for r in results:
            table.add_row(
                r["Model"], r["Split"], f"{r['rmse']:.4f}", f"{r['mae']:.4f}", f"{r['r2']:.4f}"
            )
    else:
        table.add_column("ROC-AUC", justify="right")
        table.add_column("PR-AUC", justify="right")
        table.add_column("MCC", justify="right")
        for r in results:
            table.add_row(
                r["Model"],
                r["Split"],
                f"{r['roc_auc']:.4f}",
                f"{r['pr_auc']:.4f}",
                f"{r['mcc']:.4f}",
            )

    console.print(table)


def run_kfold_benchmark(
    dataset_name: str,
    task_type: str,
    processed_dir: Path,
    n_folds: int = 5,
):
    """
    Run k-fold scaffold cross-validation for baseline models.
    Reports mean ± std metrics across folds.
    """
    import numpy as np

    data_path = processed_dir / dataset_name / "processed.csv"
    if not data_path.exists():
        log.error(f"Processed data not found for {dataset_name} at {data_path}")
        return

    log.info(f"\n[bold blue]{n_folds}-Fold Scaffold CV: {dataset_name}[/bold blue]")
    df = pd.read_csv(data_path)

    smiles_list = df["std_smiles"].tolist()
    target_cols = [c for c in df.columns if c != "std_smiles"]
    if not target_cols:
        log.error(f"No target columns found in {dataset_name}")
        return
    target = target_cols[0]

    x = batch_smiles_to_morgan(smiles_list)
    y = df[target].values

    valid_mask = ~pd.isna(y)
    x = x[valid_mask]
    y = y[valid_mask]
    smiles_valid = [smiles_list[i] for i in range(len(smiles_list)) if valid_mask[i]]

    folds = scaffold_kfold(smiles_valid, n_folds=n_folds)

    models_config = [
        ("RF", "rf", {"n_estimators": 100, "random_state": 42}),
        ("XGBoost", "xgb", {"n_estimators": 100, "random_state": 42}),
    ]

    for name, mtype, params in models_config:
        fold_metrics = []
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            model = BaselineModel(model_type=mtype, task_type=task_type, params=params)
            model.train(x[train_idx], y[train_idx])
            metrics = model.evaluate(x[val_idx], y[val_idx])
            fold_metrics.append(metrics)
            log.info(f"  Fold {fold_i + 1}: {metrics}")

        # Aggregate
        all_keys = fold_metrics[0].keys()
        summary = {}
        for key in all_keys:
            vals = [m[key] for m in fold_metrics]
            summary[key] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

        log.info(f"\n[bold green]{name} — {n_folds}-Fold Mean ± Std:[/bold green]")
        for key, val in summary.items():
            log.info(f"  {key}: {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="delaney", help="Dataset name")
    parser.add_argument(
        "--task", type=str, default="regression", choices=["regression", "classification"]
    )
    parser.add_argument(
        "--split-type", type=str, default="random_scaffold", choices=["scaffold", "random_scaffold"]
    )
    parser.add_argument(
        "--explain", action="store_true", help="Run SHAP explanation after evaluation"
    )
    parser.add_argument(
        "--kfold", type=int, default=0, help="Run k-fold scaffold CV (e.g., --kfold 5)"
    )
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    processed_path = ROOT / "data" / "processed"

    if args.kfold > 1:
        run_kfold_benchmark(
            args.dataset,
            args.task,
            processed_path,
            n_folds=args.kfold,
        )
    else:
        run_benchmark(
            args.dataset,
            args.task,
            processed_path,
            split_type=args.split_type,
            explain=args.explain,
        )
