"""Generate publication-quality figures from Phase 2A results.

Reads JSON metrics + prediction CSVs from a Phase 2A results directory and
emits PNGs into ``reports/phase3_report/figures/`` and
``reports/phase3_presentation/`` so both the IEEE-style report and the HTML
deck can pull from the same canonical figure set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


CLASSES = ["hunger", "belly_pain", "burping", "discomfort", "tiredness"]
CLASS_LABELS = ["Hunger", "Belly Pain", "Burping", "Discomfort", "Tiredness"]

FEATURE_BANKS = {
    "ast": "AST (frozen)",
    "ast_aux_adapted": "AST + aux-adapted",
    "whisper": "Whisper encoder",
    "handcrafted": "Handcrafted (411-d)",
    "ablation_no_aux_no_whisper": "AST + handcrafted",
    "ablation_no_aux_with_whisper": "AST + Whisper + handcrafted",
    "ablation_aux_no_whisper": "AST-aux + handcrafted",
    "ablation_aux_with_whisper": "AST-aux + Whisper + handcrafted",
}

CLASSIFIERS = {
    "rbf_svm_balanced": "RBF SVM (balanced)",
    "logreg_balanced": "Logistic Regression",
    "soft_voting": "Soft Voting Ensemble",
    "linear_svm_calibrated": "Linear SVM (calibrated)",
    "rf_balanced": "Random Forest",
    "prototype": "Prototype",
}

PALETTE = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#059669",
    "warn": "#D97706",
    "danger": "#DC2626",
    "muted": "#94A3B8",
    "ink": "#0F172A",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("/Users/zebra/Downloads/phase2a"))
    parser.add_argument("--report-figs", type=Path, default=Path("reports/phase3_report/figures"))
    parser.add_argument("--pres-figs", type=Path, default=Path("reports/phase3_presentation"))
    return parser.parse_args()


def save_both(fig, name: str, report_dir: Path, pres_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pres_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(report_dir / f"{name}.png")
    fig.savefig(pres_dir / f"{name}.png")
    plt.close(fig)


def load_repeated_eval(results_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for path in sorted((results_dir / "metrics").glob("phase2a_*_repeated_eval.json")):
        if path.stem.endswith("3class_dominant_repeated_eval"):
            continue
        feature_name = path.stem.removeprefix("phase2a_").removesuffix("_repeated_eval")
        feature_name = feature_name.replace("_features", "").replace("_embeddings", "")
        data = json.loads(path.read_text())
        out[feature_name] = data["summary"]
    return out


def load_3class(results_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    path = results_dir / "metrics" / "phase2a_3class_dominant_repeated_eval.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out = {}
    for feature_key, block in raw.items():
        feature_name = feature_key.replace("_features", "").replace("_embeddings", "")
        out[feature_name] = block.get("summary", {})
    return out


def load_fixed_metrics(results_dir: Path) -> dict:
    path = results_dir / "metrics" / "phase2a_all_feature_sets_metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def headline_journey_figure(report_dir: Path, pres_dir: Path) -> None:
    """Phase 1 -> Phase 2 -> Phase 2A headline progression."""
    stages = ["Phase 1\nClassical ML", "Phase 2\nHybrid Ensemble", "Phase 2A\nPretrained + RBF SVM"]
    macro_f1 = [0.270, 0.507, 0.6566]
    err = [0.0, 0.0, 0.048]
    accuracy = [0.55, 0.926, 0.7402]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(
        stages,
        macro_f1,
        yerr=err,
        color=[PALETTE["muted"], PALETTE["warn"], PALETTE["primary"]],
        edgecolor="white",
        linewidth=1.5,
        capsize=6,
        zorder=3,
    )
    for bar, value in zip(bars, macro_f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=PALETTE["ink"],
        )
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Macro-F1 (5-class strict)")
    ax.set_title("Three-phase macro-F1 progression on Donate-a-Cry / strict 5-class manifest")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.text(2, 0.78, "+143% rel. vs Phase 1\n+30% rel. vs Phase 2", ha="center", fontsize=9, color=PALETTE["accent"], fontweight="bold")
    save_both(fig, "headline_progression", report_dir, pres_dir)


def feature_bank_comparison(eval_data: dict, report_dir: Path, pres_dir: Path) -> None:
    rows = []
    for feature_key, summary in eval_data.items():
        if "rbf_svm_balanced" not in summary:
            continue
        rows.append(
            {
                "feature": FEATURE_BANKS.get(feature_key, feature_key),
                "key": feature_key,
                "mean": summary["rbf_svm_balanced"]["test_macro_f1_mean"],
                "std": summary["rbf_svm_balanced"]["test_macro_f1_std"],
            }
        )
    rows = sorted(rows, key=lambda r: r["mean"])
    labels = [r["feature"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    colors = [PALETTE["primary"] if r["key"] == "ablation_aux_with_whisper" else PALETTE["muted"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ypos = np.arange(len(labels))
    ax.barh(ypos, means, xerr=stds, color=colors, edgecolor="white", linewidth=1.2, capsize=4, zorder=3)
    for y, mean, std in zip(ypos, means, stds):
        ax.text(mean + std + 0.005, y, f"{mean:.3f} ± {std:.3f}", va="center", fontsize=9, color=PALETTE["ink"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Macro-F1 (mean ± std, 5 stratified splits)")
    ax.set_title("Phase 2A: feature-bank comparison with RBF SVM")
    ax.set_xlim(0, max(means) + max(stds) + 0.12)
    save_both(fig, "feature_bank_comparison", report_dir, pres_dir)


def classifier_comparison(eval_data: dict, report_dir: Path, pres_dir: Path) -> None:
    feature_key = "ablation_aux_with_whisper"
    if feature_key not in eval_data:
        return
    summary = eval_data[feature_key]
    rows = sorted(
        [
            {"clf": CLASSIFIERS.get(clf, clf), "key": clf, "mean": stats["test_macro_f1_mean"], "std": stats["test_macro_f1_std"]}
            for clf, stats in summary.items()
        ],
        key=lambda r: r["mean"],
    )
    labels = [r["clf"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    colors = [PALETTE["primary"] if r["key"] == "rbf_svm_balanced" else PALETTE["muted"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ypos = np.arange(len(labels))
    ax.barh(ypos, means, xerr=stds, color=colors, edgecolor="white", linewidth=1.2, capsize=4, zorder=3)
    for y, mean, std in zip(ypos, means, stds):
        ax.text(mean + std + 0.005, y, f"{mean:.3f} ± {std:.3f}", va="center", fontsize=9, color=PALETTE["ink"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Macro-F1 (mean ± std, 5 stratified splits)")
    ax.set_title("Phase 2A: classifier comparison on the best feature bank (AST-aux + Whisper + handcrafted)")
    ax.set_xlim(0, max(means) + max(stds) + 0.12)
    save_both(fig, "classifier_comparison", report_dir, pres_dir)


def best_confusion_matrix(results_dir: Path, report_dir: Path, pres_dir: Path) -> None:
    pred_path = results_dir / "predictions" / "phase2a_ablation_aux_with_whisper_rbf_svm_balanced_test_predictions.csv"
    if not pred_path.exists():
        return
    df = pd.read_csv(pred_path)
    cm = confusion_matrix(df["true_idx"], df["pred_idx"], labels=range(len(CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = LinearSegmentedColormap.from_list("blue_grad", ["#FFFFFF", PALETTE["primary"]])
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASS_LABELS, rotation=20, ha="right")
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (normalised) — best Phase 2A model")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else PALETTE["ink"]
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]*100:.0f}%", ha="center", va="center", color=color, fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="Row-normalised")
    save_both(fig, "best_confusion_matrix", report_dir, pres_dir)


def per_class_breakdown(fixed_metrics: dict, report_dir: Path, pres_dir: Path) -> None:
    stats = fixed_metrics.get("ablation_aux_with_whisper", {}).get("rbf_svm_balanced", {}).get("test", {}).get("per_class", {})
    if not stats:
        return
    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    values = np.zeros((len(metrics), len(CLASSES)))
    for j, cls in enumerate(CLASSES):
        for i, m in enumerate(metrics):
            values[i, j] = stats.get(cls, {}).get(m, 0.0)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.27
    x = np.arange(len(CLASSES))
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    for i, (label, color) in enumerate(zip(metric_labels, colors)):
        ax.bar(x + (i - 1) * width, values[i], width, label=label, color=color, edgecolor="white", zorder=3)
        for xi, v in zip(x + (i - 1) * width, values[i]):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=PALETTE["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\nn={int(stats[cls]['support'])}" for lbl, cls in zip(CLASS_LABELS, CLASSES)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Phase 2A best model — per-class precision / recall / F1 (fixed test split)")
    ax.legend(ncols=3)
    save_both(fig, "per_class_breakdown", report_dir, pres_dir)


def three_vs_five_class(eval_data: dict, three_data: dict, report_dir: Path, pres_dir: Path) -> None:
    feature_keys = list(FEATURE_BANKS.keys())
    five = []
    three = []
    labels = []
    for key in feature_keys:
        five_block = eval_data.get(key, {}).get("rbf_svm_balanced")
        three_block = three_data.get(key, {}).get("rbf_svm_balanced")
        if not (five_block and three_block):
            continue
        labels.append(FEATURE_BANKS[key])
        five.append(five_block["test_macro_f1_mean"])
        three.append(three_block["macro_f1_mean"])

    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, five, width, label="5-class strict", color=PALETTE["primary"], edgecolor="white", zorder=3)
    ax.bar(x + width / 2, three, width, label="3-class dominant", color=PALETTE["accent"], edgecolor="white", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Macro-F1 (mean, repeated splits)")
    ax.set_title("5-class strict vs 3-class dominant macro-F1 — RBF SVM across feature banks")
    ax.legend()
    save_both(fig, "three_vs_five_class", report_dir, pres_dir)


def auxiliary_training_curve(results_dir: Path, report_dir: Path, pres_dir: Path) -> None:
    path = results_dir / "auxiliary_adaptation" / "auxiliary_adaptation_metrics.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(epochs, [r["macro_f1"] for r in rows], "-o", color=PALETTE["primary"], label="Macro-F1")
    ax.plot(epochs, [r["cry_f1"] for r in rows], "-s", color=PALETTE["accent"], label="Cry F1")
    ax.plot(epochs, [r["noncry_f1"] for r in rows], "-^", color=PALETTE["warn"], label="Non-cry F1")
    ax2 = ax.twinx()
    ax2.plot(epochs, [r["train_loss"] for r in rows], ":", color=PALETTE["danger"], label="Train loss")
    ax2.set_ylabel("Train loss", color=PALETTE["danger"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["danger"])
    ax2.grid(False)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (validation cry/non-cry)")
    ax.set_ylim(0.85, 1.005)
    ax.set_title("Auxiliary cry-vs-non-cry adaptation — AST encoder fine-tune")
    ax.legend(loc="lower right")
    save_both(fig, "auxiliary_adaptation_curve", report_dir, pres_dir)


def _load_edge_results(results_dir: Path, variant: str) -> dict | None:
    repeated = results_dir / "metrics" / f"phase3_edge_student_{variant}_repeated_eval.json"
    metrics = results_dir / "metrics" / f"phase3_edge_student_{variant}_metrics.json"
    if not (repeated.exists() and metrics.exists()):
        return None
    rep = json.loads(repeated.read_text())
    met = json.loads(metrics.read_text())
    return {"repeated": rep, "metrics": met}


def edge_overview_figure(results_dir: Path, eval_data: dict, report_dir: Path, pres_dir: Path) -> None:
    """Real edge-vs-teacher footprint chart from Phase 3 distillation outputs."""
    teacher_key = "ablation_aux_with_whisper"
    teacher_summary = eval_data.get(teacher_key, {}).get("rbf_svm_balanced")
    if teacher_summary is None:
        return
    teacher_macro_f1 = teacher_summary["test_macro_f1_mean"]

    rows = []
    rows.append(
        {
            "label": "Phase 2A teacher\n(AST-aux + Whisper + handcrafted\n+ RBF SVM)",
            "params": 86_000_000,
            "size_kb": 344_000,
            "macro_f1": teacher_macro_f1,
            "latency_ms": 85.0,
            "color": "muted",
        }
    )
    for variant, color, display_label in [
        ("mn10_as", "primary", "Phase 3 student\nEfficientAT mn10_as\n(4.88M params)"),
        ("mn04_as", "accent", "Phase 3 student\nEfficientAT mn04_as\n(0.98M params)"),
    ]:
        bundle = _load_edge_results(results_dir, variant)
        if bundle is None:
            continue
        m = bundle["metrics"]
        s = bundle["repeated"]["summary"]
        rows.append(
            {
                "label": display_label,
                "params": int(m["n_params"]),
                "size_kb": float(m["int8_size_kb"]),
                "macro_f1": float(s["test_macro_f1"]["mean"]),
                "macro_f1_std": float(s["test_macro_f1"]["std"]),
                "latency_ms": float(m.get("int8_cpu_latency_ms", m.get("fp32_cpu_latency_ms", float("nan")))),
                "color": color,
            }
        )
    if len(rows) == 1:
        return

    labels = [r["label"] for r in rows]
    params = [r["params"] for r in rows]
    sizes = [r["size_kb"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    f1_errs = [r.get("macro_f1_std", 0.0) for r in rows]
    lats = [r["latency_ms"] for r in rows]
    colors = [PALETTE[r["color"]] for r in rows]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.4))
    metrics = [
        ("Parameters", params, "log", None),
        ("INT8 size (KB)", sizes, "log", None),
        ("Macro-F1 (mean ± std)", f1s, "linear", f1_errs),
        ("CPU latency (ms)", lats, "linear", None),
    ]
    for ax, (title, vals, scale, errs) in zip(axes, metrics):
        ax.bar(labels, vals, color=colors, edgecolor="white", yerr=errs, capsize=5, zorder=3)
        if scale == "log":
            ax.set_yscale(scale)
        ax.set_title(title)
        for x, v in enumerate(vals):
            label = f"{v:,.0f}" if v >= 100 else f"{v:.2f}" if v >= 1 else f"{v:.3f}"
            ax.text(x, v, label, ha="center", va="bottom", fontsize=8.5, color=PALETTE["ink"])
        ax.tick_params(axis="x", labelrotation=15, labelsize=8.5)
    fig.suptitle("Phase 3 edge distillation: keep teacher accuracy in a phone-class footprint", y=1.06, fontweight="bold")
    save_both(fig, "edge_target_overview", report_dir, pres_dir)


def edge_per_class_figure(results_dir: Path, report_dir: Path, pres_dir: Path) -> None:
    """Per-class macro-F1 retention for each EfficientAT variant."""
    variants = []
    for variant, label, color in [
        ("mn10_as", "mn10_as (4.88M)", "primary"),
        ("mn04_as", "mn04_as (0.98M)", "accent"),
    ]:
        bundle = _load_edge_results(results_dir, variant)
        if bundle:
            variants.append((variant, label, color, bundle))
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(CLASSES))
    width = 0.8 / max(1, len(variants) + 1)
    teacher_per_class = None
    for offset, (_v, label, color, bundle) in enumerate(variants):
        per_class = bundle["repeated"]["summary"]["per_class_test_f1"]
        means = [per_class[cls]["mean"] for cls in CLASSES]
        stds = [per_class[cls]["std"] for cls in CLASSES]
        ax.bar(x + (offset + 1) * width, means, width, yerr=stds, capsize=3, color=PALETTE[color], edgecolor="white", label=f"Student {label}", zorder=3)
        if teacher_per_class is None:
            teacher_test = bundle["metrics"]["best_seed_teacher_metrics"]["per_class"]
            teacher_per_class = [teacher_test[cls]["f1"] for cls in CLASSES]
    if teacher_per_class is not None:
        ax.bar(x, teacher_per_class, width, color=PALETTE["muted"], edgecolor="white", label="Teacher (best seed)", zorder=3)
    ax.set_xticks(x + width * (len(variants) / 2))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylabel("Per-class test F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 retention — Phase 2A teacher vs Phase 3 EfficientAT students")
    ax.legend(ncols=3)
    save_both(fig, "edge_per_class_retention", report_dir, pres_dir)


def confidence_distribution(results_dir: Path, report_dir: Path, pres_dir: Path) -> None:
    pred_path = results_dir / "predictions" / "phase2a_ablation_aux_with_whisper_rbf_svm_balanced_test_predictions.csv"
    if not pred_path.exists():
        return
    df = pd.read_csv(pred_path)
    correct = df["true_idx"] == df["pred_idx"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bins = np.linspace(0.2, 1.0, 17)
    ax.hist([df.loc[correct, "confidence"], df.loc[~correct, "confidence"]], bins=bins, stacked=True, color=[PALETTE["accent"], PALETTE["danger"]], label=["Correct", "Wrong"], edgecolor="white")
    ax.axvline(df["confidence"].median(), color=PALETTE["ink"], linestyle="--", alpha=0.5, label=f"Median = {df['confidence'].median():.2f}")
    ax.set_xlabel("Top-1 softmax confidence")
    ax.set_ylabel("Number of test clips")
    ax.set_title("Phase 2A best model — confidence distribution on the 204 test clips")
    ax.legend()
    save_both(fig, "confidence_distribution", report_dir, pres_dir)


def main() -> None:
    args = parse_args()
    setup_style()
    report_dir = args.report_figs
    pres_dir = args.pres_figs

    eval_data = load_repeated_eval(args.results_dir)
    three_data = load_3class(args.results_dir)
    fixed_metrics = load_fixed_metrics(args.results_dir)

    headline_journey_figure(report_dir, pres_dir)
    feature_bank_comparison(eval_data, report_dir, pres_dir)
    classifier_comparison(eval_data, report_dir, pres_dir)
    best_confusion_matrix(args.results_dir, report_dir, pres_dir)
    per_class_breakdown(fixed_metrics, report_dir, pres_dir)
    three_vs_five_class(eval_data, three_data, report_dir, pres_dir)
    auxiliary_training_curve(args.results_dir, report_dir, pres_dir)
    edge_overview_figure(args.results_dir, eval_data, report_dir, pres_dir)
    edge_per_class_figure(args.results_dir, report_dir, pres_dir)
    confidence_distribution(args.results_dir, report_dir, pres_dir)

    summary_rows = []
    for feature_key, summary in eval_data.items():
        for clf_key, stats in summary.items():
            summary_rows.append(
                {
                    "feature_bank": FEATURE_BANKS.get(feature_key, feature_key),
                    "classifier": CLASSIFIERS.get(clf_key, clf_key),
                    "macro_f1_mean": stats["test_macro_f1_mean"],
                    "macro_f1_std": stats["test_macro_f1_std"],
                    "macro_f1_min": stats["test_macro_f1_min"],
                    "macro_f1_max": stats["test_macro_f1_max"],
                }
            )
    df_summary = pd.DataFrame(summary_rows).sort_values("macro_f1_mean", ascending=False)
    df_summary.to_csv(report_dir / "repeated_eval_summary.csv", index=False)
    print(f"Wrote {len(summary_rows)} rows to {report_dir / 'repeated_eval_summary.csv'}")
    print("Top 5:")
    print(df_summary.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
