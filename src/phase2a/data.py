"""Data loading and split helpers for Phase 2A."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.phase2a.config import (
    CLASS_TO_IDX,
    CLASSES,
    MAX_DURATION_SEC,
    RANDOM_STATE,
    SAMPLE_RATE,
    SUPERVISED_MANIFEST,
    TEST_SIZE,
    VAL_SIZE,
)


def load_manifest(path: Path = SUPERVISED_MANIFEST) -> pd.DataFrame:
    """Load the strict supervised manifest and validate expected columns."""
    df = pd.read_csv(path)
    required = {"sample_id", "processed_path", "canonical_label", "source_dataset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    df = df[df["canonical_label"].isin(CLASSES)].copy()
    df["label_idx"] = df["canonical_label"].map(CLASS_TO_IDX).astype(int)
    return df.reset_index(drop=True)


def create_stratified_splits(
    df: pd.DataFrame,
    seed: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> pd.DataFrame:
    """Create a reproducible stratified train/val/test split.

    The manifest has already been globally deduplicated by exact audio hash and
    content fingerprint, so this split is safe for Phase 2A feature-bank runs.
    """
    labels = df["label_idx"].to_numpy()
    trainval_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    trainval = df.iloc[trainval_idx]
    relative_val = val_size / (1.0 - test_size)
    train_idx_rel, val_idx_rel = train_test_split(
        np.arange(len(trainval)),
        test_size=relative_val,
        random_state=seed,
        stratify=trainval["label_idx"].to_numpy(),
    )

    split = pd.Series("train", index=df.index, dtype=object)
    split.loc[df.iloc[test_idx].index] = "test"
    split.loc[trainval.iloc[val_idx_rel].index] = "val"
    out = df.copy()
    out["split"] = split
    return out


def load_or_create_splits(
    manifest_path: Path = SUPERVISED_MANIFEST,
    split_path: Path | None = None,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Load existing Phase 2A split or create it if missing."""
    if split_path is None:
        split_path = manifest_path.parent / "phase2a_split_manifest_v1.csv"
    if split_path.exists():
        return pd.read_csv(split_path)
    df = load_manifest(manifest_path)
    split_df = create_stratified_splits(df, seed=seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_path, index=False)
    write_split_summary(split_df, split_path.with_suffix(".summary.json"))
    return split_df


def write_split_summary(df: pd.DataFrame, path: Path) -> None:
    summary = {
        split: group["canonical_label"].value_counts().reindex(CLASSES, fill_value=0).to_dict()
        for split, group in df.groupby("split")
    }
    summary["total"] = df["canonical_label"].value_counts().reindex(CLASSES, fill_value=0).to_dict()
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_audio(path: str | Path, sr: int = SAMPLE_RATE, max_duration_sec: float = MAX_DURATION_SEC) -> np.ndarray:
    """Load mono audio, resampled and padded/truncated for pretrained models."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    max_len = int(sr * max_duration_sec)
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    return y.astype(np.float32)


def resolve_audio_path(path: str | Path, project_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root / p

