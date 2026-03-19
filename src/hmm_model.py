"""
hmm_model.py — Hidden Markov Model for infant cry classification (exploratory).

Like the GMM classifier, the HMM approach trains one model per class on
frame-level MFCC sequences (shape T x N_MFCC per clip).  At inference the
class whose HMM assigns the highest log-likelihood wins.

This is a *bonus* model for Phase 1 — it is not expected to be fully tuned.
"""

import numpy as np
import joblib
from pathlib import Path
from hmmlearn.hmm import GaussianHMM

from src.config import CLASSES, MODELS_DIR, RANDOM_STATE


class HMMClassifier:
    """One-vs-all sequence classifier using Gaussian HMMs."""

    def __init__(
        self,
        n_states: int = 5,
        covariance_type: str = "diag",
        n_iter: int = 100,
        random_state: int = RANDOM_STATE,
        classes: list = None,
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.classes = classes if classes is not None else CLASSES
        self.models: dict[str, GaussianHMM] = {}
        self._fitted = False

    def fit(
        self,
        sequences: list[np.ndarray],
        y: np.ndarray,
    ) -> "HMMClassifier":
        """Fit one HMM per class.

        Parameters
        ----------
        sequences : list[np.ndarray]
            Each element has shape (T_i, n_features) — variable-length
            MFCC frame sequences.
        y : np.ndarray, shape (n_samples,)
            Integer class labels aligned with *sequences*.
        """
        for idx, cls in enumerate(self.classes):
            mask = y == idx
            cls_seqs = [sequences[i] for i, m in enumerate(mask) if m]

            if not cls_seqs:
                print(f"[WARN] No sequences for class '{cls}', skipping HMM fit.")
                continue

            X_concat = np.concatenate(cls_seqs, axis=0)
            lengths = [seq.shape[0] for seq in cls_seqs]

            n_st = min(self.n_states, min(lengths))

            hmm = GaussianHMM(
                n_components=n_st,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            hmm.fit(X_concat, lengths)
            self.models[cls] = hmm
            print(f"  HMM[{cls}] fitted — {n_st} states, "
                  f"{len(cls_seqs)} sequences")

        self._fitted = True
        return self

    def predict_log_likelihood(self, sequences: list[np.ndarray]) -> np.ndarray:
        """Score each sequence under every class HMM.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        scores = np.full((len(sequences), len(self.classes)), -np.inf)

        for idx, cls in enumerate(self.classes):
            if cls not in self.models:
                continue
            hmm = self.models[cls]
            for i, seq in enumerate(sequences):
                try:
                    scores[i, idx] = hmm.score(seq)
                except ValueError:
                    scores[i, idx] = -np.inf

        return scores

    def predict(self, sequences: list[np.ndarray]) -> np.ndarray:
        """Predict class labels for a list of MFCC sequences."""
        scores = self.predict_log_likelihood(sequences)
        return np.argmax(scores, axis=1)

    def save(self, path: Path = None) -> Path:
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "hmm_classifier.joblib"
        joblib.dump(self, path)
        print(f"HMM classifier saved → {path}")
        return path

    @staticmethod
    def load(path: Path = None) -> "HMMClassifier":
        if path is None:
            path = MODELS_DIR / "hmm_classifier.joblib"
        clf = joblib.load(path)
        print(f"HMM classifier loaded ← {path}")
        return clf
