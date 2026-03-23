"""
rf_classifier.py — Random Forest classifier for infant cry audio.

Literature shows Random Forest + MFCC achieves 96.39% on 5-class infant
cry classification.  RF is inherently robust to class imbalance via
``class_weight='balanced_subsample'`` and provides feature importance
rankings for interpretability.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import CLASSES, MODELS_DIR, RANDOM_STATE


class RFClassifier:
    """Random Forest classifier with built-in scaling and feature importance."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = None,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        random_state: int = RANDOM_STATE,
        classes: list = None,
    ):
        self.classes = classes if classes is not None else CLASSES
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFClassifier":
        """Fit the Random Forest pipeline on training data."""
        self.pipeline.fit(X, y)
        self._fitted = True
        print(f"  RF fitted — {X.shape[0]} samples, {X.shape[1]} features, "
              f"{self.pipeline.named_steps['rf'].n_estimators} trees")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def feature_importances(self, feature_names: list = None) -> list[tuple]:
        """Return feature importances sorted descending.

        Returns list of (name, importance) tuples.
        """
        rf = self.pipeline.named_steps["rf"]
        importances = rf.feature_importances_
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(len(importances))]
        pairs = list(zip(feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def save(self, path: Path = None) -> Path:
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "rf_classifier.joblib"
        joblib.dump(self, path)
        print(f"RF classifier saved → {path}")
        return path

    @staticmethod
    def load(path: Path = None) -> "RFClassifier":
        if path is None:
            path = MODELS_DIR / "rf_classifier.joblib"
        clf = joblib.load(path)
        print(f"RF classifier loaded ← {path}")
        return clf
