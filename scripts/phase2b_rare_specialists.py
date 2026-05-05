"""Train rare-class one-vs-rest specialists and combine with ensemble output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2b.common import build_rbf_svm_pipeline, evaluate_probs, load_feature_bank, save_prediction_csv, split_arrays
from src.phase2b.config import FEATURES_DIR, METRICS_DIR, PREDICTIONS_DIR, RARE_CLASS_IDXS, RARE_CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, default=FEATURES_DIR / "ablation_aux_with_whisper_features.npz")
    parser.add_argument("--base-probs", type=Path, default=PREDICTIONS_DIR / "phase2b_weighted_ensemble_probs.npz")
    parser.add_argument("--output-prefix", default="phase2b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y, sample_ids, splits = load_feature_bank(args.feature_file)
    split = split_arrays(X, y, sample_ids, splits)
    base = np.load(args.base_probs, allow_pickle=True)
    val_probs = base["val_probs"].copy()
    test_probs = base["test_probs"].copy()
    val_y = base["val_y"].astype(int)
    test_y = base["test_y"].astype(int)
    val_sample_ids = base["val_sample_ids"].astype(str)
    test_sample_ids = base["test_sample_ids"].astype(str)
    assert_alignment(split["val"]["sample_ids"], split["val"]["y"], val_sample_ids, val_y, "validation")
    assert_alignment(split["test"]["sample_ids"], split["test"]["y"], test_sample_ids, test_y, "test")

    specialists = {}
    val_rare_probs = {}
    test_rare_probs = {}
    for cls_idx, cls_name in zip(RARE_CLASS_IDXS, RARE_CLASS_NAMES):
        y_train = (split["train"]["y"] == cls_idx).astype(int)
        model, config = fit_best_specialist(X.shape[1], split["train"]["X"], y_train, split["val"]["X"], (split["val"]["y"] == cls_idx).astype(int))
        model.fit(split["train"]["X"], y_train)
        specialists[cls_name] = {"model": model, "config": config}
        val_rare_probs[cls_idx] = model.predict_proba(split["val"]["X"])[:, 1]
        test_rare_probs[cls_idx] = model.predict_proba(split["test"]["X"])[:, 1]

    best = search_thresholds(val_y, val_probs, val_rare_probs)
    combined_val_probs = apply_rare_overrides(val_probs, val_rare_probs, best["thresholds"])
    combined_test_probs = apply_rare_overrides(test_probs, test_rare_probs, best["thresholds"])
    val_metrics = evaluate_probs(val_y, combined_val_probs)
    test_metrics = evaluate_probs(test_y, combined_test_probs)
    out = {
        "feature_file": str(args.feature_file),
        "base_probs": str(args.base_probs),
        "specialists": {name: {"config": value["config"]} for name, value in specialists.items()},
        "thresholds": {str(k): v for k, v in best["thresholds"].items()},
        "validation": val_metrics,
        "test": test_metrics,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{args.output_prefix}_rare_specialist_ensemble.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_prediction_csv(PREDICTIONS_DIR / f"{args.output_prefix}_rare_specialist_ensemble_val_predictions.csv", val_sample_ids, val_y, combined_val_probs)
    save_prediction_csv(PREDICTIONS_DIR / f"{args.output_prefix}_rare_specialist_ensemble_test_predictions.csv", test_sample_ids, test_y, combined_test_probs)
    print(f"Rare-specialist val macro-F1:  {val_metrics['macro_f1']:.4f}")
    print(f"Rare-specialist test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Thresholds: {out['thresholds']}")


def assert_alignment(feature_ids: np.ndarray, feature_y: np.ndarray, pred_ids: np.ndarray, pred_y: np.ndarray, split_name: str) -> None:
    if not np.array_equal(feature_ids.astype(str), pred_ids.astype(str)):
        raise ValueError(f"{split_name} sample_id mismatch between feature bank and base probabilities")
    if not np.array_equal(feature_y.astype(int), pred_y.astype(int)):
        raise ValueError(f"{split_name} label mismatch between feature bank and base probabilities")


def fit_best_specialist(n_features: int, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    configs = []
    for c_value in [1.0, 3.0, 10.0, 30.0]:
        for gamma in ["scale", 0.003, 0.01, 0.03]:
            for pca_components in [64, 128, None]:
                for use_l2 in [False, True]:
                    configs.append((c_value, gamma, pca_components, use_l2))
    best_score = -1.0
    best_config = None
    best_model = None
    for c_value, gamma, pca_components, use_l2 in configs:
        model = build_rbf_svm_pipeline(
            n_features,
            c_value=c_value,
            gamma=gamma,
            pca_components=pca_components,
            use_l2=use_l2,
            probability=True,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        best_threshold_score = max(f1_score(y_val, probs >= threshold, zero_division=0) for threshold in np.arange(0.2, 0.91, 0.05))
        if best_threshold_score > best_score:
            best_score = best_threshold_score
            best_config = {
                "C": c_value,
                "gamma": gamma,
                "pca_components": pca_components,
                "use_l2": use_l2,
                "val_binary_f1": float(best_threshold_score),
            }
            best_model = model
    return best_model, best_config


def search_thresholds(y_true: np.ndarray, base_probs: np.ndarray, rare_probs: dict[int, np.ndarray]) -> dict:
    best_score = -1.0
    best_thresholds = {idx: 0.95 for idx in rare_probs}
    grid = np.arange(0.40, 0.96, 0.05)
    for t1 in grid:
        for t2 in grid:
            thresholds = {RARE_CLASS_IDXS[0]: float(t1), RARE_CLASS_IDXS[1]: float(t2)}
            probs = apply_rare_overrides(base_probs, rare_probs, thresholds)
            score = f1_score(y_true, probs.argmax(axis=1), average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
    return {"score": best_score, "thresholds": best_thresholds}


def apply_rare_overrides(base_probs: np.ndarray, rare_probs: dict[int, np.ndarray], thresholds: dict[int, float]) -> np.ndarray:
    probs = base_probs.copy()
    for row_idx in range(len(probs)):
        hits = [(cls_idx, rare_probs[cls_idx][row_idx]) for cls_idx, threshold in thresholds.items() if rare_probs[cls_idx][row_idx] >= threshold]
        if not hits:
            continue
        cls_idx, score = max(hits, key=lambda item: item[1])
        probs[row_idx] *= 0.25
        probs[row_idx, cls_idx] = max(probs[row_idx, cls_idx], score)
        probs[row_idx] = probs[row_idx] / probs[row_idx].sum()
    return probs


if __name__ == "__main__":
    main()

