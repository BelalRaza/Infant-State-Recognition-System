"""Build validation-weighted probability ensemble from tuned Phase 2B models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2b.common import evaluate_probs, save_prediction_csv
from src.phase2b.config import METRICS_DIR, PREDICTIONS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--output-prefix", default="phase2b")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--weight-step", type=float, default=0.25)
    parser.add_argument("--random-candidates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = discover_model_names(args.predictions_dir, args.output_prefix)
    if not model_names:
        raise FileNotFoundError(f"No tuned prediction files found in {args.predictions_dir}")

    candidates = []
    for name in model_names:
        val = load_prediction(args.predictions_dir / f"{args.output_prefix}_{name}_val_predictions.csv")
        metrics = evaluate_probs(val["y"], val["probs"])
        candidates.append((name, metrics["macro_f1"]))
    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[: args.top_n]
    selected = [name for name, _ in candidates]
    print("Selected ensemble candidates:")
    for name, score in candidates:
        print(f"  {name}: val_macro_f1={score:.4f}")

    val_arrays = [load_prediction(args.predictions_dir / f"{args.output_prefix}_{name}_val_predictions.csv") for name in selected]
    test_arrays = [load_prediction(args.predictions_dir / f"{args.output_prefix}_{name}_test_predictions.csv") for name in selected]
    assert_aligned(val_arrays, "validation")
    assert_aligned(test_arrays, "test")
    weights = search_weights(val_arrays, step=args.weight_step, random_candidates=args.random_candidates, seed=args.seed)
    val_probs = weighted_average([item["probs"] for item in val_arrays], weights)
    test_probs = weighted_average([item["probs"] for item in test_arrays], weights)
    val_metrics = evaluate_probs(val_arrays[0]["y"], val_probs)
    test_metrics = evaluate_probs(test_arrays[0]["y"], test_probs)

    out = {
        "selected_models": selected,
        "weights": weights.tolist(),
        "validation": val_metrics,
        "test": test_metrics,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{args.output_prefix}_weighted_ensemble.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_prediction_csv(PREDICTIONS_DIR / f"{args.output_prefix}_weighted_ensemble_val_predictions.csv", val_arrays[0]["sample_ids"], val_arrays[0]["y"], val_probs)
    save_prediction_csv(PREDICTIONS_DIR / f"{args.output_prefix}_weighted_ensemble_test_predictions.csv", test_arrays[0]["sample_ids"], test_arrays[0]["y"], test_probs)
    np.savez_compressed(
        PREDICTIONS_DIR / f"{args.output_prefix}_weighted_ensemble_probs.npz",
        val_probs=val_probs,
        val_y=val_arrays[0]["y"],
        val_sample_ids=val_arrays[0]["sample_ids"],
        test_probs=test_probs,
        test_y=test_arrays[0]["y"],
        test_sample_ids=test_arrays[0]["sample_ids"],
        selected_models=np.asarray(selected, dtype=object),
        weights=weights,
    )
    print(f"Weighted ensemble val macro-F1:  {val_metrics['macro_f1']:.4f}")
    print(f"Weighted ensemble test macro-F1: {test_metrics['macro_f1']:.4f}")


def discover_model_names(predictions_dir: Path, prefix: str) -> list[str]:
    names = []
    for path in predictions_dir.glob(f"{prefix}_*_val_predictions.csv"):
        stem = path.name.removeprefix(f"{prefix}_").removesuffix("_val_predictions.csv")
        if stem == "weighted_ensemble":
            continue
        test_path = predictions_dir / f"{prefix}_{stem}_test_predictions.csv"
        if test_path.exists():
            names.append(stem)
    return sorted(names)


def load_prediction(path: Path) -> dict:
    df = pd.read_csv(path)
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    return {
        "sample_ids": df["sample_id"].astype(str).to_numpy(),
        "y": df["true_idx"].to_numpy(dtype=int),
        "probs": df[prob_cols].to_numpy(dtype=float),
    }


def assert_aligned(items: list[dict], name: str) -> None:
    first_ids = items[0]["sample_ids"]
    first_y = items[0]["y"]
    for idx, item in enumerate(items[1:], start=1):
        if not np.array_equal(first_ids, item["sample_ids"]):
            raise ValueError(f"{name} sample_id alignment mismatch at model index {idx}")
        if not np.array_equal(first_y, item["y"]):
            raise ValueError(f"{name} label alignment mismatch at model index {idx}")


def search_weights(items: list[dict], step: float, random_candidates: int, seed: int) -> np.ndarray:
    n = len(items)
    if n == 1:
        return np.ones(1)
    rng = np.random.default_rng(seed)
    best_score = -1.0
    best = np.ones(n) / n
    y = items[0]["y"]
    probs = [item["probs"] for item in items]

    candidate_weights = [np.ones(n) / n]
    for idx in range(n):
        one_hot = np.zeros(n)
        one_hot[idx] = 1.0
        candidate_weights.append(one_hot)
    val_scores = np.asarray([evaluate_probs(y, p)["macro_f1"] for p in probs], dtype=float)
    score_weights = np.maximum(val_scores, 1e-6)
    candidate_weights.append(score_weights / score_weights.sum())
    for alpha in [0.25, 0.5, 1.0, 2.0]:
        candidate_weights.extend(rng.dirichlet(np.ones(n) * alpha, size=random_candidates // 4))

    for weights in candidate_weights:
        score = evaluate_probs(y, weighted_average(probs, weights))["macro_f1"]
        if score > best_score:
            best_score = score
            best = weights
    return best


def weighted_average(probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack(probs, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


if __name__ == "__main__":
    main()

