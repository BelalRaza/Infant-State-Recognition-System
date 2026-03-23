"""
hmm_model.py — Hidden Markov Model for infant cry classification.

Like the GMM classifier, the HMM approach trains one model per class on
frame-level MFCC sequences (shape T x N_MFCC per clip).  At inference the
class whose HMM assigns the highest log-likelihood wins.

Improvements:
- Increased to 8 hidden states for richer temporal modeling
- Left-right topology via upper-triangular transition matrix
- Uses only first 13 MFCCs for HMM input (less noise)
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
        n_states: int = 8,
        covariance_type: str = "diag",
        n_iter: int = 150,
        n_mfcc_hmm: int = 13,
        random_state: int = RANDOM_STATE,
        classes: list = None,
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.n_mfcc_hmm = n_mfcc_hmm
        self.random_state = random_state
        self.classes = classes if classes is not None else CLASSES
        self.models: dict[str, GaussianHMM] = {}
        self._fitted = False

    def _init_left_right(self, n_states: int) -> tuple[np.ndarray, np.ndarray]:
        """Initialize left-right (Bakis) topology.

        Returns startprob and transmat for a left-right model where
        states can only stay or move forward.
        """
        startprob = np.zeros(n_states)
        startprob[0] = 1.0

        transmat = np.zeros((n_states, n_states))
        for i in range(n_states - 1):
            transmat[i, i] = 0.7      # self-loop
            transmat[i, i + 1] = 0.3  # forward
        transmat[-1, -1] = 1.0        # absorbing final state

        return startprob, transmat

    def fit(
        self,
        sequences: list[np.ndarray],
        y: np.ndarray,
    ) -> "HMMClassifier":
        """Fit one HMM per class with left-right topology.

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
            cls_seqs = [sequences[i][:, :self.n_mfcc_hmm]
                        for i, m in enumerate(mask) if m]

            if not cls_seqs:
                print(f"[WARN] No sequences for class '{cls}', skipping HMM fit.")
                continue

            X_concat = np.concatenate(cls_seqs, axis=0)
            lengths = [seq.shape[0] for seq in cls_seqs]

            n_st = min(self.n_states, min(lengths))
            startprob, transmat = self._init_left_right(n_st)

            hmm = GaussianHMM(
                n_components=n_st,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                init_params="mc",  # only init means and covariances
                params="stmc",     # train all parameters
            )
            hmm.startprob_ = startprob
            hmm.transmat_ = transmat
            hmm.fit(X_concat, lengths)
            self.models[cls] = hmm
            print(f"  HMM[{cls}] fitted — {n_st} states (left-right), "
                  f"{len(cls_seqs)} sequences, {self.n_mfcc_hmm} features")

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
                    seq_trimmed = seq[:, :self.n_mfcc_hmm]
                    scores[i, idx] = hmm.score(seq_trimmed)
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
