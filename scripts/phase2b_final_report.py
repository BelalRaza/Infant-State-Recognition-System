"""Final Phase 2B report: bootstrap CIs, per-class, source-stratified, abstention.

Reads the rare-specialist (or weighted-ensemble) test predictions from Phase 2B
and produces a single deployment-ready report containing:

  * Bootstrap 95% confidence interval on test macro-F1.
  * Per-class precision / recall / F1 with bootstrap CIs.
  * Source-stratified macro-F1 (per ``source_dataset``) so we can detect
    over-reliance on one dataset.
  * Calibration ECE (expected calibration error) using top-1 confidence.
  * Abstention curve: macro-F1 vs coverage as we drop low-confidence samples.

This is the honest "what would a clinical/product team see" view. It does not
move scores; it tells the truth about them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2a.config import CLASSES, SUPERVISED_MANIFEST
from src.phase2b.config import METRICS_DIR, PREDICTIONS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS_DIR / "phase2b_rare_specialist_ensemble_test_predictions.csv",
        help="Test prediction CSV produced by phase2b_rare_specialists.py or phase2b_weighted_ensemble.py.",
    )
    parser.add_argument("--manifest", type=Path, default=SUPERVISED_MANIFEST)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=METRICS_DIR / "phase2b_final_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    if not prob_cols:
        raise ValueError(f"No prob_* columns in {args.predictions}")
    sample_ids = df["sample_id"].astype(str).to_numpy()
    y_true = df["true_idx"].to_numpy(dtype=int)
    probs = df[prob_cols].to_numpy(dtype=float)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    manifest = pd.read_csv(args.manifest)
    source_lookup = dict(zip(manifest["sample_id"].astype(str), manifest["source_dataset"].astype(str)))
    sources = np.asarray([source_lookup.get(sid, "unknown") for sid in sample_ids])

    point = headline_metrics(y_true, y_pred)
    bootstrap = bootstrap_metrics(y_true, y_pred, n=args.n_bootstrap, seed=args.seed)
    per_class = per_class_with_ci(y_true, y_pred, n=args.n_bootstrap, seed=args.seed)
    source_metrics = source_stratified(y_true, y_pred, sources)
    ece, reliability = calibration(y_true, y_pred, confidence)
    coverage_curve = abstention_curve(y_true, y_pred, confidence)

    report = {
        "predictions_path": str(args.predictions),
        "n_samples": int(len(y_true)),
        "headline": point,
        "bootstrap_95ci": bootstrap,
        "per_class": per_class,
        "source_stratified": source_metrics,
        "calibration": {"ece": ece, "reliability_bins": reliability},
        "abstention_curve": coverage_curve,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Phase 2B Final Report ===")
    print(f"Samples: {len(y_true)}")
    print(f"Macro-F1: {point['macro_f1']:.4f}  95% CI [{bootstrap['macro_f1']['lo']:.4f}, {bootstrap['macro_f1']['hi']:.4f}]")
    print(f"Balanced acc: {point['balanced_accuracy']:.4f}  95% CI [{bootstrap['balanced_accuracy']['lo']:.4f}, {bootstrap['balanced_accuracy']['hi']:.4f}]")
    print(f"ECE (top-1): {ece:.4f}")
    print("Per-class F1 (mean [95% CI]):")
    for cls, stats in per_class.items():
        print(f"  {cls}: {stats['f1']:.4f} [{stats['f1_ci_lo']:.4f}, {stats['f1_ci_hi']:.4f}] (n={stats['support']})")
    print("Source-stratified macro-F1:")
    for src, stats in sorted(source_metrics.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {src}: macro_f1={stats['macro_f1']:.4f} balanced_acc={stats['balanced_accuracy']:.4f} (n={stats['n']})")
    print(f"Saved report: {args.output}")


def headline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, n: int, seed: int) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    rng = np.random.default_rng(seed)
    metrics = {"macro_f1": [], "weighted_f1": [], "balanced_accuracy": [], "accuracy": []}
    size = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, size, size=size)
        yt, yp = y_true[idx], y_pred[idx]
        metrics["macro_f1"].append(f1_score(yt, yp, average="macro", zero_division=0))
        metrics["weighted_f1"].append(f1_score(yt, yp, average="weighted", zero_division=0))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        metrics["accuracy"].append(accuracy_score(yt, yp))
    out = {}
    for key, values in metrics.items():
        arr = np.asarray(values)
        out[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "lo": float(np.percentile(arr, 2.5)),
            "hi": float(np.percentile(arr, 97.5)),
        }
    return out


def per_class_with_ci(y_true: np.ndarray, y_pred: np.ndarray, n: int, seed: int) -> dict:
    label_idx = list(range(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_idx, zero_division=0
    )
    rng = np.random.default_rng(seed)
    f1_samples = {idx: [] for idx in label_idx}
    size = len(y_true)
    for _ in range(n):
        boot = rng.integers(0, size, size=size)
        _, _, f1_b, _ = precision_recall_fscore_support(
            y_true[boot], y_pred[boot], labels=label_idx, zero_division=0
        )
        for idx in label_idx:
            f1_samples[idx].append(f1_b[idx])
    out = {}
    for idx, cls in enumerate(CLASSES):
        arr = np.asarray(f1_samples[idx])
        out[cls] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "f1_ci_lo": float(np.percentile(arr, 2.5)),
            "f1_ci_hi": float(np.percentile(arr, 97.5)),
        }
    return out


def source_stratified(y_true: np.ndarray, y_pred: np.ndarray, sources: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    out = {}
    for source in sorted(set(sources)):
        mask = sources == source
        if mask.sum() < 5:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        labels = sorted(set(yt))
        out[source] = {
            "n": int(mask.sum()),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)) if len(set(yt)) > 1 else float("nan"),
            "label_counts": {CLASSES[int(c)]: int((yt == c).sum()) for c in labels},
        }
    return out


def calibration(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    correct = (y_true == y_pred).astype(float)
    ece = 0.0
    total = len(y_true)
    reliability = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            reliability.append({"lo": float(lo), "hi": float(hi), "count": 0, "accuracy": None, "avg_conf": None})
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(confidence[mask].mean())
        weight = mask.sum() / total
        ece += weight * abs(acc - avg_conf)
        reliability.append({"lo": float(lo), "hi": float(hi), "count": int(mask.sum()), "accuracy": acc, "avg_conf": avg_conf})
    return float(ece), reliability


def abstention_curve(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> list[dict]:
    order = np.argsort(-confidence)
    yt = y_true[order]
    yp = y_pred[order]
    points = []
    for coverage in np.linspace(0.1, 1.0, 19):
        k = max(1, int(round(coverage * len(yt))))
        sub_yt = yt[:k]
        sub_yp = yp[:k]
        labels = sorted(set(sub_yt))
        points.append(
            {
                "coverage": float(coverage),
                "n": int(k),
                "macro_f1": float(f1_score(sub_yt, sub_yp, average="macro", labels=labels, zero_division=0)),
                "accuracy": float((sub_yt == sub_yp).mean()),
                "min_confidence": float(confidence[order][k - 1]),
            }
        )
    return points


if __name__ == "__main__":
    main()
