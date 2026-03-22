"""
data_loader.py — Load, resample, and normalise infant cry audio clips.

Provides two interfaces:

1. ``InfantCryLoader`` class (new — required by the Phase 1 rubric)
   with load_audio, load_dataset, train_val_test_split, save_split_indices.

2. Module-level helper functions ``load_dataset()`` and
   ``get_class_distribution()`` that delegate to the class.  These are
   kept for backward compatibility with ``run_pipeline.py`` and existing
   notebooks.
"""

import csv
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    SAMPLE_RATE,
    DURATION,
    N_SAMPLES,
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    RANDOM_STATE,
    TEST_SIZE,
    RESULTS_DIR,
)


class InfantCryLoader:
    """End-to-end data loading, splitting, and persistence for infant cry audio.

    Typical usage
    -------------
    >>> loader = InfantCryLoader()
    >>> dataset = loader.load_dataset("data/raw")
    >>> train, val, test = loader.train_val_test_split(dataset)
    >>> loader.save_split_indices(train, val, test, "results/splits.csv")
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        duration: float = DURATION,
        classes: list = None,
        random_state: int = RANDOM_STATE,
    ):
        self.sr = sr
        self.duration = duration
        self.classes = classes if classes is not None else CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        self.random_state = random_state

    # ── Single-file loading ───────────────────────────────────────────

    def load_audio(
        self,
        filepath: str | Path,
        sr: int = None,
        duration: float = None,
    ) -> np.ndarray:
        """Load a single WAV file, resample to *sr*, and fix its length.

        Why fixed-length?
        -----------------
        Classical ML models (GMM, SVM) operate on fixed-dimensional feature
        vectors.  If clips have different lengths, the number of MFCC frames
        differs, and the statistical summary (mean, std, ...) would be
        computed over different-length sequences — introducing a confound.
        Padding short clips with silence and truncating long ones ensures
        every clip produces an identical-shaped feature vector, which is a
        hard requirement for batch matrix operations in numpy / sklearn.

        Parameters
        ----------
        filepath : str or Path
            Path to a .wav / .mp3 / .ogg / .flac file.
        sr : int, optional
            Target sampling rate.  Defaults to ``self.sr``.
        duration : float, optional
            Target clip length in seconds.  Defaults to ``self.duration``.

        Returns
        -------
        np.ndarray
            1-D float32 waveform of length ``sr * duration``.
        """
        if sr is None:
            sr = self.sr
        if duration is None:
            duration = self.duration

        y, _ = librosa.load(filepath, sr=sr, duration=duration, mono=True)
        target_len = int(sr * duration)

        # Zero-pad if too short, truncate if too long
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            y = y[:target_len]

        return y.astype(np.float32)

    # ── Full dataset loading ──────────────────────────────────────────

    def load_dataset(
        self,
        data_dir: str | Path = None,
    ) -> list[dict]:
        """Walk *data_dir*/<class>/ and load every audio file.

        Parameters
        ----------
        data_dir : str or Path
            Root directory containing one sub-folder per class.
            Defaults to ``config.RAW_DIR``.

        Returns
        -------
        dataset : list[dict]
            Each element is::

                {
                    "audio":     np.ndarray,   # fixed-length waveform
                    "label":     str,           # class name
                    "label_idx": int,           # integer label
                    "filepath":  str,           # original file path
                }

        Corrupted or unreadable files are logged and skipped.
        """
        if data_dir is None:
            data_dir = RAW_DIR
        data_dir = Path(data_dir)

        audio_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        dataset = []
        corrupted = []

        for cls in self.classes:
            cls_dir = data_dir / cls
            if not cls_dir.is_dir():
                print(f"[WARN] Directory not found, skipping: {cls_dir}")
                continue

            audio_files = sorted(
                f for f in cls_dir.iterdir()
                if f.suffix.lower() in audio_extensions
            )
            if not audio_files:
                print(f"[WARN] No audio files in {cls_dir}")
                continue

            print(f"Loading {len(audio_files):>4d} files for class '{cls}' ...")
            for fpath in tqdm(audio_files, desc=cls, leave=False):
                try:
                    waveform = self.load_audio(fpath)
                    dataset.append({
                        "audio": waveform,
                        "label": cls,
                        "label_idx": self.class_to_idx[cls],
                        "filepath": str(fpath),
                    })
                except Exception as exc:
                    corrupted.append(str(fpath))
                    print(f"[ERROR] Could not load {fpath.name}: {exc}")

        print(f"Loaded {len(dataset)} clips across {len(self.classes)} classes.")
        if corrupted:
            print(f"[WARN] {len(corrupted)} corrupted file(s) skipped:")
            for p in corrupted:
                print(f"  - {p}")

        return dataset

    # ── Stratified splitting ──────────────────────────────────────────

    def train_val_test_split(
        self,
        dataset: list[dict],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Split the dataset into train / validation / test subsets.

        Why stratified?
        ---------------
        Our dataset is heavily imbalanced — e.g. hunger may have 382 clips
        while burping has only 8.  A naive random split could leave zero
        burping samples in the test set, making evaluation meaningless for
        that class.  Stratified splitting guarantees that every split mirrors
        the overall class proportions, so each class is represented in train,
        val, *and* test sets.  This is critical for computing per-class
        metrics (precision, recall, F1) that we use to judge model quality.

        Parameters
        ----------
        dataset : list[dict]
            Output of ``load_dataset()``.
        test_size : float
            Fraction of data reserved for testing (default 0.2 = 20 %).
        val_size : float
            Fraction of data reserved for validation (default 0.1 = 10 %).
            Taken from the *remaining* data after the test split so the
            actual fraction of the total is ``val_size * (1 - test_size)``.
        random_state : int
            Seed for reproducibility.  Defaults to ``self.random_state``.

        Returns
        -------
        train_data, val_data, test_data : list[dict]
        """
        if random_state is None:
            random_state = self.random_state

        labels = [d["label_idx"] for d in dataset]
        indices = np.arange(len(dataset))

        # First split: separate out the test set
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # Second split: separate train and validation from the remainder
        train_val_labels = [labels[i] for i in train_val_idx]
        relative_val_size = val_size / (1.0 - test_size)

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=train_val_labels,
        )

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]
        test_data = [dataset[i] for i in test_idx]

        print(f"Split sizes — train: {len(train_data)}, "
              f"val: {len(val_data)}, test: {len(test_data)}")
        return train_data, val_data, test_data

    # ── Persist split indices ─────────────────────────────────────────

    def save_split_indices(
        self,
        train_data: list[dict],
        val_data: list[dict],
        test_data: list[dict],
        save_path: str | Path = None,
    ) -> Path:
        """Save file paths and labels for each split to a CSV.

        Why save the split?
        -------------------
        In a multi-phase project (Phase 1 = classical ML, Phase 2 = DL,
        Phase 3 = hybrid) every model MUST be evaluated on the exact same
        test set.  If we re-split randomly each time, the test sets differ
        and performance comparisons become invalid — a model might look
        better simply because it got an easier test split.  Persisting the
        split to a CSV guarantees reproducibility: any script or notebook
        can reload it and get the identical partition.

        Parameters
        ----------
        train_data, val_data, test_data : list[dict]
            Outputs of ``train_val_test_split()``.
        save_path : str or Path
            Destination CSV file.  Defaults to ``results/split_indices.csv``.

        Returns
        -------
        Path
            The path to the saved CSV file.
        """
        if save_path is None:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = RESULTS_DIR / "split_indices.csv"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for split_name, split_data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            for d in split_data:
                rows.append({
                    "split": split_name,
                    "filepath": d["filepath"],
                    "label": d["label"],
                    "label_idx": d["label_idx"],
                })

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"Split indices saved -> {save_path}  ({len(df)} entries)")
        return save_path

    # ── Convenience: extract arrays from dataset dicts ────────────────

    @staticmethod
    def to_arrays(dataset: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Convert a list-of-dicts dataset to parallel numpy arrays.

        Returns
        -------
        X : np.ndarray, shape (n, sr*duration)
        y : np.ndarray, shape (n,)
        file_paths : list[str]
        """
        X = np.array([d["audio"] for d in dataset], dtype=np.float32)
        y_arr = np.array([d["label_idx"] for d in dataset], dtype=np.int64)
        fps = [d["filepath"] for d in dataset]
        return X, y_arr, fps


# =====================================================================
#  Module-level backward-compatible functions
#  (used by run_pipeline.py, notebooks, and other modules)
# =====================================================================

_default_loader = None


def _get_loader() -> InfantCryLoader:
    """Lazy-initialise a shared InfantCryLoader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = InfantCryLoader()
    return _default_loader


def load_audio(
    file_path: Path,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
) -> np.ndarray:
    """Load a single audio file — backward-compatible wrapper."""
    return _get_loader().load_audio(file_path, sr=sr, duration=duration)


def load_dataset(
    data_dir: Path = RAW_DIR,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    classes: list = None,
    return_list: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]] | tuple[list[dict], list[str]]:
    """Load entire dataset and return (X, y, file_paths) arrays.

    This is the backward-compatible wrapper used by run_pipeline.py.

    Parameters
    ----------
    return_list : bool
        If True, return (dataset_list, class_names) instead of numpy
        arrays.  This is used by standalone model scripts that feed the
        list directly into ``FeatureExtractor.extract_and_save_dataset()``.
    """
    loader = InfantCryLoader(sr=sr, duration=duration, classes=classes)
    dataset = loader.load_dataset(data_dir)
    if return_list:
        return dataset, loader.classes
    return loader.to_arrays(dataset)


def get_class_distribution(
    y: np.ndarray,
    classes: list = None,
) -> dict[str, int]:
    """Return a {class_name: count} dictionary."""
    if classes is None:
        classes = CLASSES
    unique, counts = np.unique(y, return_counts=True)
    return {classes[int(u)]: int(c) for u, c in zip(unique, counts)}
