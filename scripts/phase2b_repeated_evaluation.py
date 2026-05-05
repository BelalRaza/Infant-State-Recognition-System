"""Repeated-split evaluation of the full Phase 2B teacher pipeline.

For each random seed we:
  1. Re-stratify the supervised manifest into train / val / test.
  2. Refit each feature-bank's top-N tuned RBF SVM configs on the new train.
  3. Build a validation-weighted probability ensemble.
  4. Train rare-class specialists (belly_pain, burping) on the new train.
  5. Search rare-class thresholds on the new validation split.
  6. Score the final combined predictions on the new held-out test split.

The configs themselves are reused from a prior ``phase2b_tune_svm.py`` run so
this script is feasible on Colab. The result is a mean / std macro-F1 directly
comparable to Phase 2A's repeated-split number, which is the only way to claim
honestly that Phase 2B beats Phase 2A.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2a.data import load_manifest
from src.phase2b.common import build_rbf_svm_pipeline, evaluate_probs, load_feature_bank
from src.phase2b.config import (
    DEFAULT_FEATURE_FILES,
    FEATURES_DIR,
    METRICS_DIR,
    RARE_CLASS_IDXS,
    RARE_CLASS_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument(
        "--tuning-summary",
        type=Path,
        default=METRICS_DIR / "phase2b_svm_tuning_summary.json",
        help="Output of phase2b_tune_svm.py providing top configs per feature bank.",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--top-k-per-bank", type=int, default=2)
    parser.add_argument("--top-n-ensemble", type=int, default=8)
    parser.add_argument("--random-candidates", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=METRICS_DIR / "phase2b_repeated_evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    sample_to_label = dict(zip(manifest["sample_id"].astype(str), manifest["label_idx"].astype(int)))

    feature_banks = load_all_banks(args.features_dir)
    if not feature_banks:
        raise RuntimeError(f"No feature banks found in {args.features_dir}")

    tuning = json.loads(args.tuning_summary.read_text(encoding="utf-8"))

    rng = np.random.default_rng(args.seed)
    seeds = [int(s) for s in rng.integers(0, 10_000, size=args.n_seeds)]

    per_seed_records = []
    for seed in seeds:
        record = run_one_seed(
            seed=seed,
            sample_to_label=sample_to_label,
            feature_banks=feature_banks,
            tuning=tuning,
            top_k_per_bank=args.top_k_per_bank,
            top_n_ensemble=args.top_n_ensemble,
            random_candidates=args.random_candidates,
            test_size=args.test_size,
            val_size=args.val_size,
        )
        per_seed_records.append(record)
        print(
            f"seed={seed} ensemble_test_f1={record['ensemble_test_macro_f1']:.4f} "
            f"final_test_f1={record['final_test_macro_f1']:.4f}"
        )

    summary = summarise(per_seed_records)
    args.output.write_text(json.dumps({"seeds": seeds, "per_seed": per_seed_records, "summary": summary}, indent=2), encoding="utf-8")

    print("\n=== Phase 2B repeated-split summary ===")
    for key, stats in summary.items():
        print(f"  {key}: mean={stats['mean']:.4f} std={stats['std']:.4f} min={stats['min']:.4f} max={stats['max']:.4f}")
    print(f"Saved: {args.output}")


def load_all_banks(features_dir: Path) -> dict[str, dict]:
    banks = {}
    for name in DEFAULT_FEATURE_FILES:
        path = features_dir / name
        if not path.exists():
            continue
        feature_name = path.stem.replace("_features", "").replace("_embeddings", "")
        X, y, sample_ids, _ = load_feature_bank(path)
        order = np.argsort(sample_ids)
        banks[feature_name] = {
            "path": path,
            "X": X[order],
            "y": y[order],
            "sample_ids": sample_ids[order],
        }
    return banks


def run_one_seed(
    seed: int,
    sample_to_label: dict[str, int],
    feature_banks: dict[str, dict],
    tuning: dict,
    top_k_per_bank: int,
    top_n_ensemble: int,
    random_candidates: int,
    test_size: float,
    val_size: float,
) -> dict:
    sample_ids = sorted(sample_to_label)
    labels = np.asarray([sample_to_label[s] for s in sample_ids], dtype=int)
    train_ids, val_ids, test_ids = stratified_three_split(sample_ids, labels, test_size, val_size, seed)

    candidate_probs: dict[str, dict[str, np.ndarray]] = {}
    for feature_name, bank in feature_banks.items():
        configs = top_configs_for(tuning, feature_name, top_k_per_bank)
        if not configs:
            continue
        for rank, config in enumerate(configs, start=1):
            try:
                val_probs, test_probs, val_y, test_y, val_ids_used, test_ids_used = train_and_predict(
                    bank, config, train_ids, val_ids, test_ids
                )
            except Exception as exc:
                print(f"[seed={seed}] skip {feature_name} rank{rank}: {exc}")
                continue
            key = f"{feature_name}_rbf_rank{rank}"
            candidate_probs[key] = {
                "val_probs": val_probs,
                "test_probs": test_probs,
                "val_y": val_y,
                "test_y": test_y,
                "val_macro_f1": f1_score(val_y, val_probs.argmax(axis=1), average="macro", zero_division=0),
                "val_ids": val_ids_used,
                "test_ids": test_ids_used,
            }

    if not candidate_probs:
        raise RuntimeError(f"No candidates trained for seed {seed}")

    selected = sorted(candidate_probs.items(), key=lambda item: item[1]["val_macro_f1"], reverse=True)[:top_n_ensemble]
    selected_names = [name for name, _ in selected]

    val_y = selected[0][1]["val_y"]
    test_y = selected[0][1]["test_y"]
    val_probs_list = [item[1]["val_probs"] for item in selected]
    test_probs_list = [item[1]["test_probs"] for item in selected]

    weights = search_weights(val_probs_list, val_y, random_candidates, seed)
    ensemble_val = weighted_average(val_probs_list, weights)
    ensemble_test = weighted_average(test_probs_list, weights)

    rare_bank = pick_rare_bank(feature_banks)
    rare_val_probs, rare_test_probs, rare_configs = train_rare_specialists(rare_bank, train_ids, val_ids, test_ids)
    thresholds = search_thresholds(val_y, ensemble_val, rare_val_probs)
    final_val = apply_rare_overrides(ensemble_val, rare_val_probs, thresholds)
    final_test = apply_rare_overrides(ensemble_test, rare_test_probs, thresholds)

    ensemble_metrics = evaluate_probs(test_y, ensemble_test)
    final_metrics = evaluate_probs(test_y, final_test)

    return {
        "seed": int(seed),
        "selected_models": selected_names,
        "weights": weights.tolist(),
        "rare_specialist_configs": rare_configs,
        "thresholds": {str(k): float(v) for k, v in thresholds.items()},
        "ensemble_test_macro_f1": ensemble_metrics["macro_f1"],
        "final_test_macro_f1": final_metrics["macro_f1"],
        "ensemble_test_balanced_acc": ensemble_metrics["balanced_accuracy"],
        "final_test_balanced_acc": final_metrics["balanced_accuracy"],
        "final_test_per_class_f1": {name: stats["f1"] for name, stats in final_metrics["per_class"].items()},
    }


def stratified_three_split(sample_ids: list[str], labels: np.ndarray, test_size: float, val_size: float, seed: int):
    indices = np.arange(len(sample_ids))
    trainval_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=labels, random_state=seed)
    relative_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=relative_val, stratify=labels[trainval_idx], random_state=seed)
    ids = np.asarray(sample_ids)
    return list(ids[train_idx]), list(ids[val_idx]), list(ids[test_idx])


def top_configs_for(tuning: dict, feature_name: str, top_k: int) -> list[dict]:
    block = tuning.get(feature_name)
    if not block:
        return []
    rows = block.get("top_configs", [])
    return rows[:top_k]


def train_and_predict(bank: dict, config: dict, train_ids: list[str], val_ids: list[str], test_ids: list[str]):
    sample_ids = bank["sample_ids"]
    sid_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}
    train_idx = [sid_to_idx[s] for s in train_ids if s in sid_to_idx]
    val_idx = [sid_to_idx[s] for s in val_ids if s in sid_to_idx]
    test_idx = [sid_to_idx[s] for s in test_ids if s in sid_to_idx]
    X_train, y_train = bank["X"][train_idx], bank["y"][train_idx]
    X_val, y_val = bank["X"][val_idx], bank["y"][val_idx]
    X_test, y_test = bank["X"][test_idx], bank["y"][test_idx]

    model = build_rbf_svm_pipeline(
        bank["X"].shape[1],
        c_value=float(config["C"]),
        gamma=config["gamma"] if isinstance(config["gamma"], str) else float(config["gamma"]),
        pca_components=config["pca_components"],
        use_l2=bool(config["use_l2"]),
        probability=True,
    )
    model.fit(X_train, y_train)
    return (
        model.predict_proba(X_val),
        model.predict_proba(X_test),
        y_val,
        y_test,
        np.asarray(val_ids),
        np.asarray(test_ids),
    )


def search_weights(probs_list: list[np.ndarray], y: np.ndarray, random_candidates: int, seed: int) -> np.ndarray:
    n = len(probs_list)
    if n == 1:
        return np.ones(1)
    rng = np.random.default_rng(seed)
    candidates = [np.ones(n) / n]
    for idx in range(n):
        oh = np.zeros(n)
        oh[idx] = 1.0
        candidates.append(oh)
    val_scores = np.asarray([f1_score(y, p.argmax(axis=1), average="macro", zero_division=0) for p in probs_list])
    score_w = np.maximum(val_scores, 1e-6)
    candidates.append(score_w / score_w.sum())
    for alpha in [0.25, 0.5, 1.0, 2.0]:
        candidates.extend(rng.dirichlet(np.ones(n) * alpha, size=random_candidates // 4))

    best_score = -1.0
    best = np.ones(n) / n
    for w in candidates:
        avg = weighted_average(probs_list, w)
        score = f1_score(y, avg.argmax(axis=1), average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best = np.asarray(w)
    return best


def weighted_average(probs_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack(probs_list, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def pick_rare_bank(feature_banks: dict[str, dict]) -> dict:
    for preferred in ["ablation_aux_with_whisper", "ast_aux_adapted", "ast_tta", "ast_aux_tta", "ast"]:
        if preferred in feature_banks:
            return feature_banks[preferred]
    return next(iter(feature_banks.values()))


def train_rare_specialists(bank: dict, train_ids: list[str], val_ids: list[str], test_ids: list[str]):
    sid_to_idx = {sid: idx for idx, sid in enumerate(bank["sample_ids"])}
    train_idx = [sid_to_idx[s] for s in train_ids if s in sid_to_idx]
    val_idx = [sid_to_idx[s] for s in val_ids if s in sid_to_idx]
    test_idx = [sid_to_idx[s] for s in test_ids if s in sid_to_idx]
    X_train, y_train_full = bank["X"][train_idx], bank["y"][train_idx]
    X_val, y_val_full = bank["X"][val_idx], bank["y"][val_idx]
    X_test = bank["X"][test_idx]

    val_rare = {}
    test_rare = {}
    rare_configs = {}
    grid = [
        (c, gamma, pca, l2)
        for c in [3.0, 10.0]
        for gamma in ["scale", 0.01]
        for pca in [128, None]
        for l2 in [False, True]
    ]
    for cls_idx, cls_name in zip(RARE_CLASS_IDXS, RARE_CLASS_NAMES):
        y_train = (y_train_full == cls_idx).astype(int)
        y_val = (y_val_full == cls_idx).astype(int)
        best_score, best_model, best_cfg = -1.0, None, None
        for c, gamma, pca, l2 in grid:
            try:
                model = build_rbf_svm_pipeline(
                    bank["X"].shape[1], c_value=c, gamma=gamma, pca_components=pca, use_l2=l2, probability=True
                )
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_val)[:, 1]
                score = max(f1_score(y_val, probs >= t, zero_division=0) for t in np.arange(0.2, 0.91, 0.05))
            except Exception:
                continue
            if score > best_score:
                best_score, best_model, best_cfg = score, model, {"C": c, "gamma": gamma, "pca": pca, "l2": l2, "val_f1": float(score)}
        if best_model is None:
            continue
        val_rare[cls_idx] = best_model.predict_proba(X_val)[:, 1]
        test_rare[cls_idx] = best_model.predict_proba(X_test)[:, 1]
        rare_configs[cls_name] = best_cfg
    return val_rare, test_rare, rare_configs


def search_thresholds(y_true: np.ndarray, base_probs: np.ndarray, rare_probs: dict[int, np.ndarray]) -> dict[int, float]:
    grid = np.arange(0.40, 0.96, 0.05)
    best_score = -1.0
    best = {idx: 0.95 for idx in rare_probs}
    keys = list(rare_probs.keys())
    if not keys:
        return best
    for t1 in grid:
        if len(keys) == 1:
            thresholds = {keys[0]: float(t1)}
            avg = apply_rare_overrides(base_probs, rare_probs, thresholds)
            score = f1_score(y_true, avg.argmax(axis=1), average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best = thresholds
            continue
        for t2 in grid:
            thresholds = {keys[0]: float(t1), keys[1]: float(t2)}
            avg = apply_rare_overrides(base_probs, rare_probs, thresholds)
            score = f1_score(y_true, avg.argmax(axis=1), average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best = thresholds
    return best


def apply_rare_overrides(base_probs: np.ndarray, rare_probs: dict[int, np.ndarray], thresholds: dict[int, float]) -> np.ndarray:
    if not rare_probs:
        return base_probs
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


def summarise(records: list[dict]) -> dict:
    keys = [
        "ensemble_test_macro_f1",
        "final_test_macro_f1",
        "ensemble_test_balanced_acc",
        "final_test_balanced_acc",
    ]
    out = {}
    for key in keys:
        values = [r[key] for r in records]
        out[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    per_class_records: dict[str, list[float]] = {}
    for record in records:
        for cls, value in record["final_test_per_class_f1"].items():
            per_class_records.setdefault(cls, []).append(value)
    out["per_class_final_f1"] = {
        cls: {"mean": float(np.mean(vals)), "std": float(np.std(vals))} for cls, vals in per_class_records.items()
    }
    return out


if __name__ == "__main__":
    main()
