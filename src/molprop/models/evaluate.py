"""
Evaluation visualization module.

Generates publication-quality plots for model evaluation:
PR curves, ROC curves, calibration diagrams, regression scatter, error histograms.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task: str = "classification",
    threshold: float = 0.5,
) -> dict:
    """
    Compute a comprehensive dict of evaluation metrics.

    Args:
        y_true: Ground-truth labels or values.
        y_score: Model output scores (probabilities for classification, raw values for regression).
        task: ``'classification'`` or ``'regression'``.
        threshold: Decision threshold for binary classification metrics.

    Returns:
        Dictionary mapping metric name to float value.
    """
    metrics: dict = {}

    if task == "classification":
        y_pred = (y_score >= threshold).astype(int)
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        metrics["avg_precision"] = round(float(average_precision_score(y_true, y_score)), 4)
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
        metrics["f1"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        metrics["mcc"] = round(float(matthews_corrcoef(y_true, y_pred)), 4)
        tn, fp_count, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        denom = tn + fp_count
        metrics["specificity"] = round(float(tn / denom) if denom > 0 else 0.0, 4)
        metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y_true, y_pred)), 4)
    elif task == "regression":
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_score))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_true, y_score)), 4)
        metrics["r2"] = round(float(r2_score(y_true, y_score)), 4)
        residuals = y_true - y_score
        metrics["mean_error"] = round(float(residuals.mean()), 4)
        metrics["std_error"] = round(float(residuals.std()), 4)
        r, _ = pearsonr(y_true, y_score)
        metrics["pearson_r"] = round(float(r), 4)
    else:
        raise ValueError(f"Unknown task type: {task!r}. Use 'classification' or 'regression'.")

    return metrics


# Styling
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": "#dee2e6",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
    }
)

COLORS = {
    "primary": "#4361ee",
    "secondary": "#7209b7",
    "accent": "#f72585",
    "success": "#06d6a0",
    "neutral": "#6c757d",
}


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """Precision-Recall curve with AP score annotation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ap = average_precision_score(y_true, y_score)
    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_score,
        ax=ax,
        color=COLORS["primary"],
        name=f"{model_name} (AP={ap:.3f})",
    )
    ax.set_title(f"Precision-Recall Curve — {model_name} on {dataset_name}", fontsize=13)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    path = output_dir / f"pr_curve_{model_name}_{dataset_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"PR curve saved to {path}")
    return path


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """ROC curve with AUC annotation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    auc = roc_auc_score(y_true, y_score)
    RocCurveDisplay.from_predictions(
        y_true,
        y_score,
        ax=ax,
        color=COLORS["secondary"],
        name=f"{model_name} (AUC={auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "--", color=COLORS["neutral"], alpha=0.5, label="Random")
    ax.set_title(f"ROC Curve — {model_name} on {dataset_name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    path = output_dir / f"roc_curve_{model_name}_{dataset_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"ROC curve saved to {path}")
    return path


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    n_bins: int = 10,
) -> Path:
    """Reliability diagram (calibration curve)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    ax.plot(
        prob_pred,
        prob_true,
        "s-",
        color=COLORS["primary"],
        label=model_name,
        markersize=6,
    )
    ax.plot([0, 1], [0, 1], "--", color=COLORS["neutral"], alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name} on {dataset_name}", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()

    path = output_dir / f"calibration_{model_name}_{dataset_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Calibration curve saved to {path}")
    return path


def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """Predicted vs actual scatter plot with RMSE and R² annotation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.5,
        s=20,
        color=COLORS["primary"],
        edgecolors="white",
        linewidth=0.3,
    )

    # Perfect prediction line
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "--", color=COLORS["accent"], alpha=0.7, label="Perfect")

    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(f"Predicted vs Actual — {model_name} on {dataset_name}", fontsize=13)

    # Metrics annotation
    textstr = f"RMSE = {rmse:.3f}\nR² = {r2:.3f}"
    props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#dee2e6")
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    path = output_dir / f"scatter_{model_name}_{dataset_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Scatter plot saved to {path}")
    return path


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
) -> Path:
    """Histogram of prediction residuals."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        residuals,
        bins=30,
        color=COLORS["primary"],
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0, color=COLORS["accent"], linestyle="--", linewidth=1.5, label="Zero error")
    ax.set_xlabel("Residual (Actual - Predicted)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Error Distribution — {model_name} on {dataset_name}", fontsize=13)

    textstr = f"MAE = {mae:.3f}\nStd = {residuals.std():.3f}"
    props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#dee2e6")
    ax.text(
        0.95,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = output_dir / f"errors_{model_name}_{dataset_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Error distribution saved to {path}")
    return path
