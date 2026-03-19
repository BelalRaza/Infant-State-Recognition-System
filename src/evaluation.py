"""
evaluation.py — Scoring, reporting, and plotting utilities.

Every model in this project calls the same evaluation functions so that
metrics are consistent and comparable.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import CLASSES, RESULTS_DIR, PLOTS_DIR, METRICS_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list = None) -> dict:
    """Return a dictionary of standard classification metrics.

    Keys: accuracy, macro_f1, weighted_f1, per_class (dict of P/R/F1).
    """
    if classes is None:
        classes = CLASSES

    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
            for cls in classes
            if cls in report
        },
    }
    return metrics


def print_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list = None) -> None:
    """Pretty-print the sklearn classification report."""
    if classes is None:
        classes = CLASSES
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))


def save_metrics(metrics: dict, model_name: str, output_dir: Path = METRICS_DIR) -> Path:
    """Write metrics dict to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {path}")
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list = None,
    model_name: str = "model",
    normalize: bool = True,
    save: bool = True,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Plot (and optionally save) a confusion matrix heatmap."""
    if classes is None:
        classes = CLASSES

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = f"{model_name} — Normalised Confusion Matrix"
    else:
        fmt = "d"
        title = f"{model_name} — Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_confusion_matrix.png"
        fig.savefig(path, dpi=150)
        print(f"Plot saved → {path}")

    plt.close(fig)


def plot_class_f1(
    metrics: dict,
    model_name: str = "model",
    save: bool = True,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Bar chart of per-class F1 scores."""
    per_class = metrics.get("per_class", {})
    classes = list(per_class.keys())
    f1_scores = [per_class[c]["f1"] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, f1_scores, color="steelblue", edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{model_name} — Per-Class F1")

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.2f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_f1_per_class.png"
        fig.savefig(path, dpi=150)
        print(f"Plot saved → {path}")

    plt.close(fig)


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    classes: list = None,
) -> dict:
    """Run all evaluation steps: print report, compute & save metrics, plot CM."""
    if classes is None:
        classes = CLASSES

    print(f"\n{'='*60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*60}\n")

    print_report(y_true, y_pred, classes)
    metrics = compute_metrics(y_true, y_pred, classes)
    save_metrics(metrics, model_name)
    plot_confusion_matrix(y_true, y_pred, classes, model_name)
    plot_class_f1(metrics, model_name)

    print(f"\nAccuracy   : {metrics['accuracy']:.4f}")
    print(f"Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    return metrics
