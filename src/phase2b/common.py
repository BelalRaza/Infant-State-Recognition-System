"""Shared Phase 2B helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler
from sklearn.svm import SVC

from src.phase2a.config import CLASSES


def load_feature_bank(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return (
        data["X"].astype(np.float32),
        data["y"].astype(np.int64),
        data["sample_ids"].astype(str),
        data["splits"].astype(str),
    )


def split_arrays(X: np.ndarray, y: np.ndarray, sample_ids: np.ndarray, splits: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for split in ["train", "val", "test"]:
        mask = splits == split
        out[split] = {"X": X[mask], "y": y[mask], "sample_ids": sample_ids[mask]}
    return out


def build_rbf_svm_pipeline(
    n_features: int,
    c_value: float = 3.0,
    gamma: str | float = "scale",
    pca_components: int | None = 128,
    use_l2: bool = False,
    probability: bool = False,
) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if pca_components is not None and pca_components < n_features:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
    if use_l2:
        steps.append(("l2", Normalizer()))
    steps.append(
        (
            "clf",
            SVC(
                kernel="rbf",
                C=c_value,
                gamma=gamma,
                class_weight="balanced",
                probability=probability,
                random_state=42,
            ),
        )
    )
    return Pipeline(steps)


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    report = classification_report(
        y_true,
        preds,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASSES
        },
        "confusion_matrix": confusion_matrix(y_true, preds, labels=list(range(len(CLASSES)))).tolist(),
    }


def save_prediction_csv(path: Path, sample_ids: np.ndarray, y_true: np.ndarray, probs: np.ndarray) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_idx": y_true,
            "pred_idx": probs.argmax(axis=1),
            "confidence": probs.max(axis=1),
        }
    )
    for idx, name in enumerate(CLASSES):
        df[f"prob_{name}"] = probs[:, idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

