"""Repeated-split evaluation for Phase 2A cached feature banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2a.classifiers import PrefitSoftVoting, build_models, evaluate_predictions, load_npz
from src.phase2a.config import METRICS_DIR, RANDOM_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--feature-name", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=METRICS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y, sample_ids, _splits = load_npz(args.feature_file)
    feature_name = args.feature_name or args.feature_file.stem.replace("_embeddings", "").replace("_features", "")
    all_results = []

    for repeat in range(args.repeats):
        seed = RANDOM_STATE + repeat
        trainval_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=args.test_size,
            random_state=seed,
            stratify=y,
        )
        relative_val = args.val_size / (1.0 - args.test_size)
        train_idx_rel, val_idx_rel = train_test_split(
            np.arange(len(trainval_idx)),
            test_size=relative_val,
            random_state=seed,
            stratify=y[trainval_idx],
        )
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]

        repeat_result = {"repeat": repeat, "seed": seed, "models": {}}
        trained: dict[str, object] = {}
        for model_name, model in build_models(X.shape[1]).items():
            model.fit(X[train_idx], y[train_idx])
            trained[model_name] = model
            val_probs = model.predict_proba(X[val_idx])
            test_probs = model.predict_proba(X[test_idx])
            repeat_result["models"][model_name] = {
                "validation": evaluate_predictions(y[val_idx], val_probs),
                "test": evaluate_predictions(y[test_idx], test_probs),
            }
        ensemble_estimators = [(name, model) for name, model in trained.items() if name != "prototype"]
        if len(ensemble_estimators) >= 2:
            ensemble = PrefitSoftVoting(ensemble_estimators)
            repeat_result["models"]["soft_voting"] = {
                "validation": evaluate_predictions(y[val_idx], ensemble.predict_proba(X[val_idx])),
                "test": evaluate_predictions(y[test_idx], ensemble.predict_proba(X[test_idx])),
            }
        all_results.append(repeat_result)
        best = max(
            repeat_result["models"].items(),
            key=lambda item: item[1]["test"]["macro_f1"],
        )
        print(f"repeat={repeat} best={best[0]} test_macro_f1={best[1]['test']['macro_f1']:.4f}")

    summary = summarize(all_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"phase2a_{feature_name}_repeated_eval.json"
    out_path.write_text(json.dumps({"runs": all_results, "summary": summary}, indent=2), encoding="utf-8")
    print(f"Saved repeated evaluation: {out_path}")
    print(json.dumps(summary, indent=2))


def summarize(results: list[dict]) -> dict:
    model_names = sorted(results[0]["models"].keys()) if results else []
    summary = {}
    for name in model_names:
        vals = np.asarray([run["models"][name]["test"]["macro_f1"] for run in results], dtype=float)
        summary[name] = {
            "test_macro_f1_mean": float(vals.mean()),
            "test_macro_f1_std": float(vals.std(ddof=0)),
            "test_macro_f1_min": float(vals.min()),
            "test_macro_f1_max": float(vals.max()),
        }
    return summary


if __name__ == "__main__":
    main()

