"""
evaluation.py — Scoring, reporting, and plotting utilities.

Every model in this project calls the same evaluation functions so that
metrics are consistent and comparable.

Includes: accuracy, macro/weighted F1, AUC-ROC, MCC, Cohen's Kappa,
confusion matrices, per-class F1 charts, t-SNE visualization, and
feature importance plots.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

from src.config import CLASSES, NUM_CLASSES, RESULTS_DIR, PLOTS_DIR, METRICS_DIR


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    classes: list = None,
) -> dict:
    """Return a dictionary of classification metrics.

    Keys: accuracy, macro_f1, weighted_f1, mcc, kappa, auc_roc, per_class.
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
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
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

    # AUC-ROC (one-vs-rest, macro) — requires probability estimates
    if y_proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(len(classes))))
            metrics["auc_roc"] = float(roc_auc_score(
                y_bin, y_proba, average="macro", multi_class="ovr",
            ))
        except Exception:
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None

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
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)  # avoid division by zero
        cm = cm.astype(float) / row_sums
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


def plot_model_comparison(
    all_metrics: dict,
    save: bool = True,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Bar chart comparing macro F1 across all models."""
    model_names = list(all_metrics.keys())
    macro_f1s = [all_metrics[m]["macro_f1"] for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax.bar(model_names, macro_f1s, color=colors, edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Model Comparison — Macro F1")

    for bar, score in zip(bars, macro_f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "model_comparison_macro_f1.png"
        fig.savefig(path, dpi=150)
        print(f"Plot saved → {path}")

    plt.close(fig)


def plot_tsne(
    X: np.ndarray,
    y: np.ndarray,
    classes: list = None,
    save: bool = True,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """t-SNE visualization of the feature space colored by class."""
    if classes is None:
        classes = CLASSES

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(y) - 1))
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, cls in enumerate(classes):
        mask = y == idx
        if mask.sum() > 0:
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                label=cls, alpha=0.6, s=30,
            )

    ax.legend(title="Class", loc="best")
    ax.set_title("t-SNE Feature Space Visualization")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "tsne_feature_space.png"
        fig.savefig(path, dpi=150)
        print(f"Plot saved → {path}")

    plt.close(fig)


def plot_feature_importance(
    importances: list[tuple],
    model_name: str = "model",
    top_n: int = 20,
    save: bool = True,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Horizontal bar chart of top-N feature importances."""
    top = importances[:top_n]
    names = [t[0] for t in top][::-1]
    values = [t[1] for t in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names, values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances")
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_feature_importance.png"
        fig.savefig(path, dpi=150)
        print(f"Plot saved → {path}")

    plt.close(fig)


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    y_proba: np.ndarray = None,
    classes: list = None,
) -> dict:
    """Run all evaluation steps: print report, compute & save metrics, plot CM."""
    if classes is None:
        classes = CLASSES

    print(f"\n{'='*60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*60}\n")

    print_report(y_true, y_pred, classes)
    metrics = compute_metrics(y_true, y_pred, y_proba, classes)
    save_metrics(metrics, model_name)
    plot_confusion_matrix(y_true, y_pred, classes, model_name)
    plot_class_f1(metrics, model_name)

    print(f"\nAccuracy   : {metrics['accuracy']:.4f}")
    print(f"Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"MCC        : {metrics['mcc']:.4f}")
    print(f"Kappa      : {metrics['kappa']:.4f}")
    if metrics['auc_roc'] is not None:
        print(f"AUC-ROC    : {metrics['auc_roc']:.4f}")
    return metrics
