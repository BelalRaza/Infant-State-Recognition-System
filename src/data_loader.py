"""
data_loader.py — Load, resample, and normalise infant cry audio clips.

Each class is stored in its own subdirectory under data/raw/.  This module
walks the tree, loads every .wav file, pads or truncates to a fixed duration,
and returns parallel arrays of waveforms + integer labels.
"""

import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    SAMPLE_RATE,
    DURATION,
    N_SAMPLES,
    CLASSES,
    CLASS_TO_IDX,
)


def load_audio(file_path: Path, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """Load a single audio file, resample, and fix length.

    Parameters
    ----------
    file_path : Path
        Absolute or relative path to the .wav / .mp3 / .ogg file.
    sr : int
        Target sampling rate.
    duration : float
        Target duration in seconds.  Shorter clips are zero-padded;
        longer clips are truncated from the right.

    Returns
    -------
    np.ndarray
        1-D float32 waveform of length ``sr * duration``.
    """
    y, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
    target_len = int(sr * duration)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]

    return y.astype(np.float32)


def load_dataset(
    data_dir: Path = RAW_DIR,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    classes: list = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Walk *data_dir*/<class>/ and load every audio file.

    Parameters
    ----------
    data_dir : Path
        Root directory containing one sub-folder per class.
    sr : int
        Target sampling rate.
    duration : float
        Fixed clip length in seconds.
    classes : list[str] | None
        Subset of classes to load.  Defaults to ``config.CLASSES``.

    Returns
    -------
    X : np.ndarray, shape (n_samples, sr*duration)
        Waveform matrix.
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    file_paths : list[str]
        Original file paths (useful for debugging).
    """
    if classes is None:
        classes = CLASSES

    waveforms = []
    labels = []
    file_paths = []
    audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

    for cls in classes:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            print(f"[WARN] Directory not found, skipping: {cls_dir}")
            continue

        audio_files = sorted(
            f for f in cls_dir.iterdir() if f.suffix.lower() in audio_extensions
        )
        if not audio_files:
            print(f"[WARN] No audio files in {cls_dir}")
            continue

        print(f"Loading {len(audio_files):>4d} files for class '{cls}' ...")
        for fpath in tqdm(audio_files, desc=cls, leave=False):
            try:
                y = load_audio(fpath, sr=sr, duration=duration)
                waveforms.append(y)
                labels.append(CLASS_TO_IDX[cls])
                file_paths.append(str(fpath))
            except Exception as exc:
                print(f"[ERROR] Could not load {fpath}: {exc}")

    X = np.array(waveforms, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"Loaded {len(X)} clips across {len(classes)} classes.")
    return X, y, file_paths


def get_class_distribution(y: np.ndarray, classes: list = None) -> dict[str, int]:
    """Return a {class_name: count} dictionary."""
    if classes is None:
        classes = CLASSES
    unique, counts = np.unique(y, return_counts=True)
    return {classes[int(u)]: int(c) for u, c in zip(unique, counts)}
