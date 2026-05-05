"""Safe auxiliary adaptation utilities for weak infant-audio data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.phase2a.config import (
    AUX_APPROVED_SPLITS,
    AUX_EXCLUDED_LABELS,
    CLASSES,
    RANDOM_STATE,
    UNIFIED_MANIFEST,
)
from src.phase2a.data import load_audio, resolve_audio_path

CRY_LABELS = set(CLASSES) | {"cry_unlabeled"}
NON_CRY_PREFIXES = ("non_cry",)


def binary_aux_label(canonical_label: str) -> int | None:
    """Map approved weak labels to cry-vs-noncry targets."""
    if canonical_label in AUX_EXCLUDED_LABELS:
        return None
    if canonical_label in CRY_LABELS:
        return 1
    if canonical_label.startswith(NON_CRY_PREFIXES):
        return 0
    return None


def load_auxiliary_manifest(
    manifest_path: Path = UNIFIED_MANIFEST,
    include_external_train: bool = True,
) -> pd.DataFrame:
    """Load rows that are safe for auxiliary cry-domain adaptation.

    This never uses rows marked for final 5-class evaluation, and it does not
    promote weak/unlabeled rows into five-class labels.
    """
    df = pd.read_csv(manifest_path)
    if include_external_train:
        mask = (
            (df["include_for_aux_pretraining"] == True)
            | ((df["split"].isin(AUX_APPROVED_SPLITS)) & (df["include_for_eval"] == False))
        )
    else:
        mask = df["include_for_aux_pretraining"] == True
    safe = df[mask & (df["dedupe_status"] == "kept") & (df["include_for_eval"] == False)].copy()
    safe["binary_label"] = safe["canonical_label"].map(binary_aux_label)
    safe = safe.dropna(subset=["binary_label"]).copy()
    safe["binary_label"] = safe["binary_label"].astype(int)
    return safe.reset_index(drop=True)


def split_auxiliary_manifest(df: pd.DataFrame, val_size: float = 0.15, seed: int = RANDOM_STATE) -> pd.DataFrame:
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        random_state=seed,
        stratify=df["binary_label"].to_numpy(),
    )
    out = df.copy()
    out["aux_split"] = "train"
    out.loc[val_idx, "aux_split"] = "val"
    return out


class AuxiliaryAudioDataset(Dataset):
    """Dataset for weak cry-vs-noncry adaptation."""

    def __init__(self, manifest: pd.DataFrame, project_root: Path):
        self.manifest = manifest.reset_index(drop=True)
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        audio_path = resolve_audio_path(row["processed_path"], self.project_root)
        audio = load_audio(audio_path)
        return {
            "audio": audio,
            "label": int(row["binary_label"]),
            "sample_id": str(row["sample_id"]),
        }


def make_ast_collate_fn(processor):
    """Create a collate function that converts waveforms to AST inputs."""

    def collate(batch: list[dict]) -> dict:
        audio = [item["audio"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        inputs["labels"] = labels
        return inputs

    return collate

