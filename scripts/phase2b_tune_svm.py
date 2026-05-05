"""Tune RBF SVMs on cached Phase 2A/2B feature banks."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2b.common import build_rbf_svm_pipeline, evaluate_probs, load_feature_bank, save_prediction_csv, split_arrays
from src.phase2b.config import (
    ARTIFACTS_DIR,
    DEFAULT_FEATURE_FILES,
    FEATURES_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    SVM_C_VALUES,
    SVM_GAMMAS,
    SVM_PCA_COMPONENTS,
    SVM_USE_L2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--output-prefix", default="phase2b")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-configs", type=int, default=0, help="0 means try all configs.")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for name in DEFAULT_FEATURE_FILES:
        path = args.features_dir / name
        if not path.exists():
            print(f"skip missing feature bank: {name}")
            continue
        feature_name = path.stem.replace("_features", "").replace("_embeddings", "")
        print(f"\n=== Tuning {feature_name} ===")
        result = tune_one_feature_bank(path, feature_name, args.output_prefix, args.top_k, args.max_configs, args.cv_folds, args.cv_repeats)
        all_results[feature_name] = result

    summary_path = METRICS_DIR / f"{args.output_prefix}_svm_tuning_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved tuning summary: {summary_path}")


def tune_one_feature_bank(path: Path, feature_name: str, output_prefix: str, top_k: int, max_configs: int, cv_folds: int, cv_repeats: int) -> dict:
    X, y, sample_ids, splits = load_feature_bank(path)
    split = split_arrays(X, y, sample_ids, splits)
    configs = list(product(SVM_PCA_COMPONENTS, SVM_C_VALUES, SVM_GAMMAS, SVM_USE_L2))
    if max_configs > 0:
        configs = configs[:max_configs]

    scored = []
    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=42)
    for pca_components, c_value, gamma, use_l2 in configs:
        fold_scores = []
        X_train_all = split["train"]["X"]
        y_train_all = split["train"]["y"]
        for train_idx, holdout_idx in cv.split(X_train_all, y_train_all):
            model = build_rbf_svm_pipeline(
                X.shape[1],
                c_value=c_value,
                gamma=gamma,
                pca_components=pca_components,
                use_l2=use_l2,
                probability=False,
            )
            model.fit(X_train_all[train_idx], y_train_all[train_idx])
            preds = model.predict(X_train_all[holdout_idx])
            fold_scores.append(f1_score(y_train_all[holdout_idx], preds, average="macro", zero_division=0))
        scored.append(
            {
                "pca_components": pca_components,
                "C": c_value,
                "gamma": gamma,
                "use_l2": use_l2,
                "cv_macro_f1_mean": float(np.mean(fold_scores)),
                "cv_macro_f1_std": float(np.std(fold_scores)),
                "cv_folds": cv_folds,
                "cv_repeats": cv_repeats,
            }
        )

    scored = sorted(scored, key=lambda row: row["cv_macro_f1_mean"], reverse=True)
    best_rows = scored[:top_k]
    final_models = {}
    final_results = {"feature_path": str(path), "searched": len(scored), "top_configs": best_rows, "final_models": {}}

    for rank, row in enumerate(best_rows, start=1):
        model = build_rbf_svm_pipeline(
            X.shape[1],
            c_value=row["C"],
            gamma=row["gamma"],
            pca_components=row["pca_components"],
            use_l2=row["use_l2"],
            probability=True,
        )
        model.fit(split["train"]["X"], split["train"]["y"])
        val_probs = model.predict_proba(split["val"]["X"])
        test_probs = model.predict_proba(split["test"]["X"])
        model_name = f"{feature_name}_rbf_rank{rank}"
        final_models[model_name] = model
        final_results["final_models"][model_name] = {
            "config": row,
            "validation": evaluate_probs(split["val"]["y"], val_probs),
            "test": evaluate_probs(split["test"]["y"], test_probs),
        }
        save_prediction_csv(
            PREDICTIONS_DIR / f"{output_prefix}_{model_name}_val_predictions.csv",
            split["val"]["sample_ids"],
            split["val"]["y"],
            val_probs,
        )
        save_prediction_csv(
            PREDICTIONS_DIR / f"{output_prefix}_{model_name}_test_predictions.csv",
            split["test"]["sample_ids"],
            split["test"]["y"],
            test_probs,
        )
        print(
            f"{model_name}: val={final_results['final_models'][model_name]['validation']['macro_f1']:.4f} "
            f"test={final_results['final_models'][model_name]['test']['macro_f1']:.4f}"
        )

    joblib.dump(
        {"feature_path": str(path), "feature_name": feature_name, "models": final_models, "results": final_results},
        ARTIFACTS_DIR / f"{output_prefix}_{feature_name}_tuned_rbf_svms.joblib",
    )
    (METRICS_DIR / f"{output_prefix}_{feature_name}_svm_tuning.json").write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    return final_results

if __name__ == "__main__":
    main()

