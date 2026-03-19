"""
feature_extraction.py — Extract MFCC and statistical audio features.

Two levels of features are produced:

1. **Frame-level** — raw MFCC matrix per clip (used by HMM).
2. **Clip-level** — statistical summary of MFCCs plus spectral descriptors
   (used by GMM and SVM).
"""

import librosa
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path
from tqdm import tqdm

from src.config import (
    SAMPLE_RATE,
    N_MFCC,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    FEATURES_DIR,
)


# ── Frame-level features ──────────────────────────────────────────────


def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute MFCCs for a single waveform.

    Returns
    -------
    np.ndarray, shape (N_MFCC, T)
        MFCC matrix where T is the number of time frames.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH,
        n_fft=N_FFT, n_mels=N_MELS,
    )
    return mfcc


def extract_delta(mfcc: np.ndarray, order: int = 1) -> np.ndarray:
    """Compute delta (velocity) or delta-delta (acceleration) of MFCCs."""
    return librosa.feature.delta(mfcc, order=order)


# ── Clip-level statistical features ───────────────────────────────────


def summarise_frames(matrix: np.ndarray) -> np.ndarray:
    """Reduce a (n_features, T) matrix to a 1-D vector via statistics.

    For each row (feature), compute: mean, std, min, max, skew, kurtosis.
    Output length = n_features * 6.
    """
    funcs = [
        np.mean,
        np.std,
        np.min,
        np.max,
        lambda row: float(sp_stats.skew(row)),
        lambda row: float(sp_stats.kurtosis(row)),
    ]
    summary = []
    for row in matrix:
        summary.extend(f(row) for f in funcs)
    return np.array(summary, dtype=np.float32)


def extract_spectral_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute global spectral descriptors for a clip.

    Returns a 1-D vector of summary statistics for:
    - Zero-crossing rate
    - Spectral centroid
    - Spectral bandwidth
    - Spectral rolloff
    - RMS energy
    """
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    descriptors = np.vstack([zcr, centroid, bandwidth, rolloff, rms])
    return summarise_frames(descriptors)


def extract_clip_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Full clip-level feature vector (MFCC stats + delta stats + spectral).

    Components
    ----------
    1. MFCC summary         : N_MFCC * 6 = 240
    2. Delta-MFCC summary   : N_MFCC * 6 = 240
    3. Delta2-MFCC summary  : N_MFCC * 6 = 240
    4. Spectral descriptors : 5 * 6       =  30
    ─────────────────────────────────────────
    Total                                   750
    """
    mfcc = extract_mfcc(y, sr)
    delta1 = extract_delta(mfcc, order=1)
    delta2 = extract_delta(mfcc, order=2)

    mfcc_stats = summarise_frames(mfcc)
    delta1_stats = summarise_frames(delta1)
    delta2_stats = summarise_frames(delta2)
    spectral = extract_spectral_features(y, sr)

    return np.concatenate([mfcc_stats, delta1_stats, delta2_stats, spectral])


# ── Batch extraction ──────────────────────────────────────────────────


def extract_features_batch(
    X: np.ndarray,
    sr: int = SAMPLE_RATE,
    return_frame_level: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    """Extract clip-level features for every waveform in *X*.

    Parameters
    ----------
    X : np.ndarray, shape (n_clips, n_samples)
    sr : int
    return_frame_level : bool
        If True, also return a list of raw MFCC matrices (for HMM).

    Returns
    -------
    features : np.ndarray, shape (n_clips, feature_dim)
    mfcc_sequences : list[np.ndarray]  (only when *return_frame_level* is True)
    """
    clip_features = []
    mfcc_sequences = []

    for waveform in tqdm(X, desc="Extracting features"):
        clip_features.append(extract_clip_features(waveform, sr))
        if return_frame_level:
            mfcc_sequences.append(extract_mfcc(waveform, sr).T)  # (T, N_MFCC)

    features = np.array(clip_features, dtype=np.float32)

    if return_frame_level:
        return features, mfcc_sequences
    return features


def save_features(
    features: np.ndarray,
    labels: np.ndarray,
    tag: str = "train",
    output_dir: Path = FEATURES_DIR,
) -> None:
    """Persist feature matrix and labels as .npz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{tag}_features.npz"
    np.savez_compressed(path, X=features, y=labels)
    print(f"Saved features → {path}  (shape {features.shape})")


def load_features(tag: str = "train", input_dir: Path = FEATURES_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load a previously saved .npz feature file."""
    path = input_dir / f"{tag}_features.npz"
    data = np.load(path)
    return data["X"], data["y"]
