"""
xgb_classifier.py — XGBoost (Gradient Boosting) classifier for infant cry audio.

XGBoost is a strong gradient boosting implementation that handles class
imbalance via sample weighting and often achieves state-of-the-art results
on tabular/feature-based classification tasks.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import CLASSES, MODELS_DIR, RANDOM_STATE, NUM_CLASSES


class XGBCryClassifier:
    """XGBoost classifier with built-in scaling and class balancing."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = RANDOM_STATE,
        classes: list = None,
    ):
        self.classes = classes if classes is not None else CLASSES
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBCryClassifier":
        """Fit XGBoost with balanced sample weights."""
        X_scaled = self.scaler.fit_transform(X)
        sample_weights = compute_sample_weight("balanced", y)
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        self._fitted = True
        print(f"  XGBoost fitted — {X.shape[0]} samples, {X.shape[1]} features")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def feature_importances(self, feature_names: list = None) -> list[tuple]:
        """Return feature importances sorted descending."""
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(len(importances))]
        pairs = list(zip(feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def save(self, path: Path = None) -> Path:
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "xgb_classifier.joblib"
        joblib.dump(self, path)
        print(f"XGBoost classifier saved → {path}")
        return path

    @staticmethod
    def load(path: Path = None) -> "XGBCryClassifier":
        if path is None:
            path = MODELS_DIR / "xgb_classifier.joblib"
        clf = joblib.load(path)
        print(f"XGBoost classifier loaded ← {path}")
        return clf
