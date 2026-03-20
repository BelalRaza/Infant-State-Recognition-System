"""
svm_classifier.py — Support Vector Machine classifier for infant cry audio.

Uses an RBF-kernel SVM with grid-searched hyper-parameters (C, gamma).
Feature scaling is handled internally via a pipeline so that the caller
only needs to pass raw feature matrices.

Layer 2 imbalance handling: ``class_weight='balanced'`` tells the SVM to
scale the penalty parameter C inversely proportional to class frequency,
so minority-class errors cost more.  This works together with the audio
augmentation layer (Layer 1) to handle residual imbalance.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.config import CLASSES, MODELS_DIR, RANDOM_STATE


# Default hyper-parameter grid for GridSearchCV
DEFAULT_PARAM_GRID = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", "auto", 0.01, 0.001],
    "svm__kernel": ["rbf"],
}


class SVMClassifier:
    """Wrapper around sklearn SVC with built-in scaling and grid search."""

    def __init__(
        self,
        C: float = 10.0,
        gamma: str = "scale",
        kernel: str = "rbf",
        random_state: int = RANDOM_STATE,
        classes: list = None,
    ):
        self.classes = classes if classes is not None else CLASSES
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=C,
                gamma=gamma,
                kernel=kernel,
                class_weight="balanced",
                random_state=random_state,
                decision_function_shape="ovr",
                probability=True,
            )),
        ])
        self._fitted = False
        self.best_params: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """Fit the SVM pipeline on training data."""
        self.pipeline.fit(X, y)
        self._fitted = True
        print(f"  SVM fitted — {X.shape[0]} samples, {X.shape[1]} features")
        return self

    def fit_with_grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: dict = None,
        cv: int = 5,
        scoring: str = "f1_macro",
        n_jobs: int = -1,
    ) -> "SVMClassifier":
        """Fit with exhaustive grid search over hyper-parameters.

        Parameters
        ----------
        X, y : training data
        param_grid : dict  — keys must use the ``pipeline__step`` prefix
        cv : int  — number of cross-validation folds
        scoring : str  — metric to optimise
        n_jobs : int  — parallelism (-1 = all cores)
        """
        if param_grid is None:
            param_grid = DEFAULT_PARAM_GRID

        grid = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            refit=True,
        )
        grid.fit(X, y)

        self.pipeline = grid.best_estimator_
        self.best_params = grid.best_params_
        self._fitted = True

        print(f"  SVM grid search complete — best params: {self.best_params}")
        print(f"  Best CV {scoring}: {grid.best_score_:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (requires probability=True)."""
        return self.pipeline.predict_proba(X)

    def save(self, path: Path = None) -> Path:
        """Serialise the full classifier."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "svm_classifier.joblib"
        joblib.dump(self, path)
        print(f"SVM classifier saved → {path}")
        return path

    @staticmethod
    def load(path: Path = None) -> "SVMClassifier":
        """Load a previously saved classifier."""
        if path is None:
            path = MODELS_DIR / "svm_classifier.joblib"
        clf = joblib.load(path)
        print(f"SVM classifier loaded ← {path}")
        return clf
