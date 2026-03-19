"""
gmm_classifier.py — Gaussian Mixture Model classifier for infant cry audio.

Strategy: train one GMM per class on that class's feature vectors.  At
inference time, score the test sample under every class GMM and predict
the class whose model assigns the highest log-likelihood.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.config import CLASSES, MODELS_DIR, RANDOM_STATE


class GMMClassifier:
    """One-vs-all density-based classifier using Gaussian Mixture Models."""

    def __init__(
        self,
        n_components: int = 8,
        covariance_type: str = "diag",
        max_iter: int = 200,
        random_state: int = RANDOM_STATE,
        classes: list = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.classes = classes if classes is not None else CLASSES
        self.models: dict[str, GaussianMixture] = {}
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GMMClassifier":
        """Fit one GMM per class.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)  — integer labels
        """
        X_scaled = self.scaler.fit_transform(X)

        for idx, cls in enumerate(self.classes):
            mask = y == idx
            if mask.sum() == 0:
                print(f"[WARN] No samples for class '{cls}', skipping GMM fit.")
                continue

            X_cls = X_scaled[mask]
            n_comp = min(self.n_components, X_cls.shape[0])

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_init=3,
            )
            gmm.fit(X_cls)
            self.models[cls] = gmm
            print(f"  GMM[{cls}] fitted — {n_comp} components, "
                  f"{X_cls.shape[0]} samples")

        self._fitted = True
        return self

    def predict_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Return log-likelihood matrix, shape (n_samples, n_classes)."""
        X_scaled = self.scaler.transform(X)
        scores = np.zeros((X_scaled.shape[0], len(self.classes)))

        for idx, cls in enumerate(self.classes):
            if cls in self.models:
                scores[:, idx] = self.models[cls].score_samples(X_scaled)
            else:
                scores[:, idx] = -np.inf

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*."""
        scores = self.predict_log_likelihood(X)
        return np.argmax(scores, axis=1)

    def save(self, path: Path = None) -> Path:
        """Serialise the full classifier (scaler + all GMMs)."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "gmm_classifier.joblib"
        joblib.dump(self, path)
        print(f"GMM classifier saved → {path}")
        return path

    @staticmethod
    def load(path: Path = None) -> "GMMClassifier":
        """Load a previously saved classifier."""
        if path is None:
            path = MODELS_DIR / "gmm_classifier.joblib"
        clf = joblib.load(path)
        print(f"GMM classifier loaded ← {path}")
        return clf
