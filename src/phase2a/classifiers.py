"""Classical/few-shot classifiers for Phase 2A embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC

from src.phase2a.config import (
    ABSTAIN_THRESHOLDS,
    ARTIFACTS_DIR,
    CLASSES,
    METRICS_DIR,
    PCA_COMPONENTS,
    PREDICTIONS_DIR,
)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["sample_ids"], data["splits"]


class PrototypeClassifier:
    """SimpleShot-style cosine nearest prototype classifier."""

    def __init__(self):
        self.classes_: np.ndarray | None = None
        self.prototypes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xn = _l2_normalize(X)
        self.classes_ = np.unique(y)
        self.prototypes_ = np.vstack([Xn[y == cls].mean(axis=0) for cls in self.classes_])
        self.prototypes_ = _l2_normalize(self.prototypes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.prototypes_ is None:
            raise RuntimeError("PrototypeClassifier is not fitted.")
        sims = _l2_normalize(X) @ self.prototypes_.T
        probs = _softmax(sims * 10.0)
        full = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
        for col, cls in enumerate(self.classes_):
            full[:, int(cls)] = probs[:, col]
        return full

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


class PrefitSoftVoting:
    """Average predict_proba of already-fitted estimators without refitting."""

    def __init__(self, fitted_models: list[tuple[str, object]]):
        if not fitted_models:
            raise ValueError("PrefitSoftVoting requires at least one fitted model.")
        self.fitted_models = fitted_models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.stack([model.predict_proba(X) for _, model in self.fitted_models], axis=0)
        return probs.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return X / denom


def _softmax(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=1, keepdims=True)


def build_models(n_features: int) -> dict[str, object]:
    pca_components = [n for n in PCA_COMPONENTS if n < n_features]
    pca_n = max(pca_components) if pca_components else min(n_features, 32)
    return {
        "prototype": PrototypeClassifier(),
        "logreg_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_n, random_state=42)),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)),
            ]
        ),
        "linear_svm_calibrated": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_n, random_state=42)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced", C=0.5, max_iter=5000, dual="auto"),
                        cv=3,
                    ),
                ),
            ]
        ),
        "rbf_svm_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_n, random_state=42)),
                ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, C=3.0, gamma="scale")),
            ]
        ),
        "rf_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_n, random_state=42)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    report = classification_report(y_true, preds, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in CLASSES
        },
        "confusion_matrix": confusion_matrix(y_true, preds, labels=list(range(len(CLASSES)))).tolist(),
        "abstention": abstention_metrics(y_true, probs),
    }
    return metrics


def abstention_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    out = {}
    for threshold in ABSTAIN_THRESHOLDS:
        keep = confidence >= threshold
        if keep.sum() == 0:
            out[str(threshold)] = {"coverage": 0.0, "accuracy": None, "macro_f1": None}
        else:
            out[str(threshold)] = {
                "coverage": float(keep.mean()),
                "accuracy": float(accuracy_score(y_true[keep], preds[keep])),
                "macro_f1": float(f1_score(y_true[keep], preds[keep], average="macro", zero_division=0)),
            }
    return out


def train_and_evaluate_feature_set(
    feature_path: Path,
    feature_name: str,
    output_prefix: str = "phase2a",
    artifacts_dir: Path = ARTIFACTS_DIR,
    metrics_dir: Path = METRICS_DIR,
    predictions_dir: Path = PREDICTIONS_DIR,
) -> dict:
    X, y, sample_ids, splits = load_npz(feature_path)
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    trained = {}
    for name, model in build_models(X.shape[1]).items():
        model.fit(X_train, y_train)
        trained[name] = model
        val_probs = model.predict_proba(X_val)
        test_probs = model.predict_proba(X_test)
        results[name] = {
            "validation": evaluate_predictions(y_val, val_probs),
            "test": evaluate_predictions(y_test, test_probs),
        }
        _write_predictions(predictions_dir / f"{output_prefix}_{feature_name}_{name}_test_predictions.csv", sample_ids[test_mask], y_test, test_probs)

    ensemble = _build_prefit_soft_voting(trained)
    if ensemble is not None:
        val_probs = ensemble.predict_proba(X_val)
        test_probs = ensemble.predict_proba(X_test)
        results["soft_voting"] = {
            "validation": evaluate_predictions(y_val, val_probs),
            "test": evaluate_predictions(y_test, test_probs),
        }
        trained["soft_voting"] = ensemble
        _write_predictions(predictions_dir / f"{output_prefix}_{feature_name}_soft_voting_test_predictions.csv", sample_ids[test_mask], y_test, test_probs)

    joblib.dump(trained, artifacts_dir / f"{output_prefix}_{feature_name}_models.joblib")
    metrics_path = metrics_dir / f"{output_prefix}_{feature_name}_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _build_prefit_soft_voting(models: dict[str, object]) -> PrefitSoftVoting | None:
    estimators = [(name, model) for name, model in models.items() if name != "prototype"]
    if len(estimators) < 2:
        return None
    return PrefitSoftVoting(estimators)


def _write_predictions(path: Path, sample_ids: np.ndarray, y_true: np.ndarray, probs: np.ndarray) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        probs.argmax(axis=1),
        labels=list(range(len(CLASSES))),
        zero_division=0,
    )
    _ = (precision, recall, f1, support)
    df = pd.DataFrame({"sample_id": sample_ids, "true_idx": y_true, "pred_idx": probs.argmax(axis=1), "confidence": probs.max(axis=1)})
    for idx, cls in enumerate(CLASSES):
        df[f"prob_{cls}"] = probs[:, idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

