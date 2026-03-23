"""
ensemble_classifier.py — Stacking ensemble that combines multiple classifiers.

Stacks probability outputs from GMM, SVM, RF, and XGBoost into a
meta-learner (Logistic Regression) to produce the best combined model.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

from src.config import CLASSES, MODELS_DIR, RANDOM_STATE, NUM_CLASSES


class EnsembleClassifier:
    """Stacking ensemble that combines base model probability predictions."""

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        classes: list = None,
    ):
        self.classes = classes if classes is not None else CLASSES
        self.random_state = random_state
        self.meta_scaler = StandardScaler()
        self.meta_learner = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            multi_class="multinomial",
        )
        self.base_models = {}
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        base_models: dict,
    ) -> "EnsembleClassifier":
        """Fit the stacking ensemble.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        base_models : dict
            Mapping of model_name -> fitted model instance.
            Each model must have a predict_proba(X) method that returns
            (n_samples, n_classes) probability arrays.
        """
        self.base_models = base_models

        # Generate stacked meta-features from base model probabilities
        meta_features = self._get_meta_features(X_train)

        # Fit meta-learner on stacked probabilities
        meta_scaled = self.meta_scaler.fit_transform(meta_features)
        self.meta_learner.fit(meta_scaled, y_train)
        self._fitted = True

        print(f"  Ensemble fitted — {len(base_models)} base models, "
              f"meta-feature dim = {meta_features.shape[1]}")
        return self

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Stack probability outputs from all base models."""
        proba_list = []
        for name, model in self.base_models.items():
            try:
                proba = model.predict_proba(X)
                proba_list.append(proba)
            except Exception as exc:
                print(f"  [WARN] {name} predict_proba failed: {exc}")
                # Fall back to zeros
                proba_list.append(np.zeros((X.shape[0], NUM_CLASSES)))
        return np.hstack(proba_list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_features = self._get_meta_features(X)
        meta_scaled = self.meta_scaler.transform(meta_features)
        return self.meta_learner.predict(meta_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_features = self._get_meta_features(X)
        meta_scaled = self.meta_scaler.transform(meta_features)
        return self.meta_learner.predict_proba(meta_scaled)

    def save(self, path: Path = None) -> Path:
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "ensemble_classifier.joblib"
        joblib.dump(self, path)
        print(f"Ensemble classifier saved → {path}")
        return path

    @staticmethod
    def load(path: Path = None) -> "EnsembleClassifier":
        if path is None:
            path = MODELS_DIR / "ensemble_classifier.joblib"
        clf = joblib.load(path)
        print(f"Ensemble classifier loaded ← {path}")
        return clf
