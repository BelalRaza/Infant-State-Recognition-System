"""Phase 2 evaluation utilities."""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
)

from src.phase2.config import CLASSES, IDX_TO_CLASS, NUM_CLASSES, PLOTS_DIR, METRICS_DIR


def evaluate_model(model, dataloader, device, model_name="Model"):
    """Run full evaluation and return metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for mel, domain, labels in dataloader:
            mel = mel.to(device)
            domain = domain.to(device)
            logits = model(mel, domain)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(all_labels, all_preds)),
        "kappa": float(cohen_kappa_score(all_labels, all_preds)),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro",
        ))
    except Exception:
        metrics["auc_roc"] = None

    all_class_labels = list(range(NUM_CLASSES))
    report = classification_report(
        all_labels, all_preds, labels=all_class_labels,
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    metrics["per_class"] = {}
    for cls_name in CLASSES:
        if cls_name in report:
            metrics["per_class"][cls_name] = {
                "precision": report[cls_name]["precision"],
                "recall": report[cls_name]["recall"],
                "f1": report[cls_name]["f1-score"],
                "support": report[cls_name]["support"],
            }

    print(f"\n{'='*60}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, labels=all_class_labels,
                                target_names=CLASSES, zero_division=0))
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"  MCC:       {metrics['mcc']:.4f}")
    auc_str = f"{metrics['auc_roc']:.4f}" if metrics["auc_roc"] else "N/A"
    print(f"  AUC-ROC:   {auc_str}")

    return metrics, all_preds, all_labels, all_probs


def plot_confusion_matrix(labels, preds, model_name, save_dir=None):
    """Plot and save confusion matrix."""
    if save_dir is None:
        save_dir = PLOTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    path = save_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")
    return path


def plot_training_history(history, model_name, save_dir=None):
    """Plot training curves."""
    if save_dir is None:
        save_dir = PLOTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_f1"], label="Train")
    axes[1].plot(history["val_f1"], label="Val")
    axes[1].set_title("Macro F1")
    axes[1].legend()

    axes[2].plot(history["lr"])
    axes[2].set_title("Learning Rate")
    axes[2].set_yscale("log")

    fig.suptitle(f"{model_name} Training History", fontsize=14)
    plt.tight_layout()
    path = save_dir / f"{model_name}_training_history.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path}")
    return path


def save_metrics(metrics, model_name, save_dir=None):
    """Save metrics to JSON."""
    if save_dir is None:
        save_dir = METRICS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / f"{model_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics saved: {path}")
    return path


def plot_ablation_comparison(all_results, save_dir=None):
    """Bar chart comparing macro F1 across ablation variants."""
    if save_dir is None:
        save_dir = PLOTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    names = list(all_results.keys())
    f1s = [all_results[n]["macro_f1"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ecc71" if n == "full_model" else "#3498db" for n in names]
    bars = ax.bar(range(len(names)), f1s, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Ablation Study — Macro F1 Comparison")

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = save_dir / "ablation_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Ablation plot saved: {path}")
    return path


def make_ablation_table(all_results, phase1_baseline=0.2703):
    """Print a formatted ablation table."""
    print(f"\n{'='*80}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    print(f"  Phase 1 ML baseline (SVM): MacF1 = {phase1_baseline:.4f}")
    print(f"\n  {'Variant':<20s} {'Acc':>7s} {'MacF1':>7s} {'WgtF1':>7s} "
          f"{'MCC':>7s} {'AUC':>7s} {'Delta':>7s}")
    print("  " + "-" * 62)

    best_name = max(all_results, key=lambda n: all_results[n]["macro_f1"])
    for name, m in all_results.items():
        auc = f"{m['auc_roc']:.4f}" if m.get("auc_roc") else "  N/A"
        delta = m["macro_f1"] - phase1_baseline
        marker = " <--" if name == best_name else ""
        print(f"  {name:<20s} {m['accuracy']:>7.4f} {m['macro_f1']:>7.4f} "
              f"{m['weighted_f1']:>7.4f} {m['mcc']:>7.4f} {auc:>7s} "
              f"{delta:>+7.4f}{marker}")


def three_model_comparison(ml_metrics, dl_metrics, hybrid_metrics):
    """Print the mandatory ML vs DL vs Hybrid comparison table."""
    print(f"\n{'='*70}")
    print("  MANDATORY ABLATION: ML vs DL vs Hybrid")
    print(f"{'='*70}")
    print(f"  {'Model':<25s} {'Acc':>7s} {'MacF1':>7s} {'WgtF1':>7s} {'MCC':>7s}")
    print("  " + "-" * 50)

    for name, m in [("A: ML Baseline (SVM)", ml_metrics),
                    ("B: DL Only", dl_metrics),
                    ("C: Hybrid ML+DL", hybrid_metrics)]:
        print(f"  {name:<25s} {m['accuracy']:>7.4f} {m['macro_f1']:>7.4f} "
              f"{m['weighted_f1']:>7.4f} {m['mcc']:>7.4f}")
