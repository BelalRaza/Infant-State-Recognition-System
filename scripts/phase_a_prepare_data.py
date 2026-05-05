#!/usr/bin/env python3
"""Prepare the Phase A data layer for infant cry classification.

This script is intentionally manifest-first:
- scans the legacy data/raw/<class>/ tree,
- separates originals from existing _aug_ files,
- writes canonical 16 kHz mono audio,
- computes basic quality metrics,
- creates leakage-aware split manifests,
- optionally extracts current 411-d handcrafted features.

It does not download external datasets. External datasets should be added as
new source folders and registered in the manifest before model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSES, SAMPLE_RATE as LEGACY_SAMPLE_RATE  # noqa: E402
from src.feature_extractor import FeatureExtractor  # noqa: E402


CANONICAL_SAMPLE_RATE = 16_000
SOURCE_DATASET = "donateacry_local"
PROCESSING_VERSION = "v1_16k_mono"
SPLIT_VERSION = "split_v1_local_originals"
LABEL_CONFIDENCE_BY_SOURCE = {
    SOURCE_DATASET: 0.55,  # Parent-reported Donate-a-Cry-style labels.
}


@dataclass
class SampleRecord:
    sample_id: str
    source_dataset: str
    source_file: str
    raw_path: str
    processed_path: str
    raw_sha256: str
    audio_sha256_16k: str
    source_label: str
    canonical_label: str
    task_tags: str
    infant_id: str
    infant_id_confidence: str
    parent_sample_key: str
    dedupe_cluster_id: str
    is_augmented: bool
    augmentation_tag: str
    duration_sec_raw: float
    duration_sec_processed: float
    sample_rate_raw: int
    sample_rate_processed: int
    rms: float
    peak: float
    clipped_fraction: float
    silence_fraction: float
    zero_crossing_rate: float
    spectral_centroid_mean: float
    technical_quality: float
    label_confidence: float
    leakage_risk: str
    usable_for_training: bool
    usable_for_eval: bool
    exclusion_reason: str
    processing_version: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_audio(audio: np.ndarray) -> str:
    arr = np.asarray(audio, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def parse_augmented(stem: str) -> tuple[bool, str, str]:
    if "_aug_" not in stem:
        return False, stem, ""
    parent, tag = stem.split("_aug_", 1)
    return True, parent, tag


def parse_infant_id(stem: str) -> tuple[str, str]:
    parent_stem = stem.split("_aug_", 1)[0]
    maybe_uuid = parent_stem[:36].lower()
    uuid_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    if uuid_re.match(maybe_uuid):
        return maybe_uuid, "parsed"
    return "unknown", "unknown"


def iter_audio_files(raw_dir: Path) -> Iterable[tuple[str, Path]]:
    audio_exts = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    for cls in CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.is_dir():
            continue
        for path in sorted(cls_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in audio_exts:
                yield cls, path


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int, float]:
    raw_info = sf.info(str(path))
    raw_sr = int(raw_info.samplerate)
    raw_duration = float(raw_info.frames / raw_info.samplerate) if raw_info.samplerate else 0.0
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), raw_sr, raw_duration


def compute_quality(audio: np.ndarray, sr: int) -> dict[str, float]:
    if len(audio) == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "clipped_fraction": 1.0,
            "silence_fraction": 1.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_mean": 0.0,
            "technical_quality": 0.0,
        }

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    clipped_fraction = float(np.mean(np.abs(audio) >= 0.999))
    silence_fraction = float(np.mean(np.abs(audio) < 1e-4))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))

    score = 1.0
    if len(audio) / sr < 0.75:
        score -= 0.35
    if rms < 0.005:
        score -= 0.25
    if peak < 0.02:
        score -= 0.20
    if clipped_fraction > 0.01:
        score -= 0.20
    if silence_fraction > 0.80:
        score -= 0.20
    if not math.isfinite(centroid) or centroid <= 0:
        score -= 0.10

    return {
        "rms": rms,
        "peak": peak,
        "clipped_fraction": clipped_fraction,
        "silence_fraction": silence_fraction,
        "zero_crossing_rate": zcr,
        "spectral_centroid_mean": centroid if math.isfinite(centroid) else 0.0,
        "technical_quality": float(np.clip(score, 0.0, 1.0)),
    }


def quality_exclusion(metrics: dict[str, float], load_error: str = "") -> str:
    if load_error:
        return f"load_error: {load_error}"
    if metrics["technical_quality"] < 0.50:
        return "technical_quality_below_0.50"
    return ""


def prepare_manifest(raw_dir: Path, data_lake: Path) -> pd.DataFrame:
    processed_root = data_lake / "processed" / PROCESSING_VERSION / SOURCE_DATASET
    processed_root.mkdir(parents=True, exist_ok=True)
    rows: list[SampleRecord] = []
    counters: Counter[str] = Counter()

    files = list(iter_audio_files(raw_dir))
    for canonical_label, raw_path in tqdm(files, desc="Preparing audio"):
        counters[canonical_label] += 1
        sample_id = f"{SOURCE_DATASET}_{canonical_label}_{counters[canonical_label]:06d}"
        is_augmented, parent_stem, aug_tag = parse_augmented(raw_path.stem)
        infant_id, infant_conf = parse_infant_id(raw_path.stem)
        parent_key = f"{SOURCE_DATASET}:{canonical_label}:{parent_stem}"
        dedupe_cluster_id = hashlib.sha1(parent_key.encode("utf-8")).hexdigest()[:16]
        out_dir = processed_root / canonical_label
        out_dir.mkdir(parents=True, exist_ok=True)
        processed_path = out_dir / f"{safe_name(sample_id)}.wav"
        raw_sha = sha256_file(raw_path)
        load_error = ""

        try:
            audio, raw_sr, raw_duration = load_audio(raw_path, CANONICAL_SAMPLE_RATE)
            if len(audio) > 0:
                sf.write(str(processed_path), audio, CANONICAL_SAMPLE_RATE)
            audio_sha = sha256_audio(audio)
            duration_processed = len(audio) / CANONICAL_SAMPLE_RATE
            metrics = compute_quality(audio, CANONICAL_SAMPLE_RATE)
        except Exception as exc:  # keep the row so failures are auditable
            load_error = str(exc)
            raw_sr = 0
            raw_duration = 0.0
            duration_processed = 0.0
            audio_sha = ""
            metrics = compute_quality(np.array([], dtype=np.float32), CANONICAL_SAMPLE_RATE)
            processed_path = Path("")

        exclusion = quality_exclusion(metrics, load_error)
        usable_for_eval = (not is_augmented) and not exclusion
        usable_for_training = not exclusion
        leakage_risk = "medium" if infant_id != "unknown" else "high"
        if is_augmented:
            leakage_risk = "critical_for_eval"

        rows.append(
            SampleRecord(
                sample_id=sample_id,
                source_dataset=SOURCE_DATASET,
                source_file=raw_path.name,
                raw_path=str(raw_path.relative_to(PROJECT_ROOT)),
                processed_path=str(processed_path.relative_to(PROJECT_ROOT)) if processed_path else "",
                raw_sha256=raw_sha,
                audio_sha256_16k=audio_sha,
                source_label=canonical_label,
                canonical_label=canonical_label,
                task_tags="cause_5class",
                infant_id=infant_id,
                infant_id_confidence=infant_conf,
                parent_sample_key=parent_key,
                dedupe_cluster_id=dedupe_cluster_id,
                is_augmented=is_augmented,
                augmentation_tag=aug_tag,
                duration_sec_raw=raw_duration,
                duration_sec_processed=duration_processed,
                sample_rate_raw=raw_sr,
                sample_rate_processed=CANONICAL_SAMPLE_RATE,
                rms=metrics["rms"],
                peak=metrics["peak"],
                clipped_fraction=metrics["clipped_fraction"],
                silence_fraction=metrics["silence_fraction"],
                zero_crossing_rate=metrics["zero_crossing_rate"],
                spectral_centroid_mean=metrics["spectral_centroid_mean"],
                technical_quality=metrics["technical_quality"],
                label_confidence=LABEL_CONFIDENCE_BY_SOURCE[SOURCE_DATASET],
                leakage_risk=leakage_risk,
                usable_for_training=usable_for_training,
                usable_for_eval=usable_for_eval,
                exclusion_reason=exclusion,
                processing_version=PROCESSING_VERSION,
            )
        )

    manifest = pd.DataFrame([asdict(r) for r in rows])
    manifest_dir = data_lake / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_dir / "raw_manifest.csv", index=False)
    manifest.to_csv(manifest_dir / "processed_manifest.csv", index=False)
    return manifest


def write_dedupe_report(manifest: pd.DataFrame, data_lake: Path) -> pd.DataFrame:
    exact_clusters = defaultdict(list)
    for row in manifest.itertuples(index=False):
        if row.audio_sha256_16k:
            exact_clusters[row.audio_sha256_16k].append(row.sample_id)

    cluster_rows = []
    for row in manifest.itertuples(index=False):
        exact_size = len(exact_clusters.get(row.audio_sha256_16k, []))
        cluster_rows.append(
            {
                "sample_id": row.sample_id,
                "dedupe_cluster_id": row.dedupe_cluster_id,
                "parent_sample_key": row.parent_sample_key,
                "audio_sha256_16k": row.audio_sha256_16k,
                "exact_audio_duplicate_count": exact_size,
                "is_augmented": row.is_augmented,
            }
        )
    dedupe = pd.DataFrame(cluster_rows)
    out_path = data_lake / "manifests" / "dedupe_clusters.csv"
    dedupe.to_csv(out_path, index=False)
    return dedupe


def create_splits(manifest: pd.DataFrame, data_lake: Path, seed: int) -> pd.DataFrame:
    originals = manifest[
        (~manifest["is_augmented"])
        & (manifest["usable_for_eval"])
        & (manifest["canonical_label"].isin(CLASSES))
    ].copy()

    if originals.empty:
        raise RuntimeError("No usable original samples found for splitting.")

    y = originals["canonical_label"].to_numpy()
    idx = np.arange(len(originals))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    temp_y = y[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        stratify=temp_y,
    )

    split_by_parent = {}
    for split_name, split_indices in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        for i in split_indices:
            split_by_parent[originals.iloc[i]["parent_sample_key"]] = split_name

    split_rows = []
    for row in manifest.itertuples(index=False):
        split = split_by_parent.get(row.parent_sample_key, "excluded")
        reason = row.exclusion_reason
        include_for_training = False
        include_for_eval = False

        if row.is_augmented:
            if split == "train" and row.usable_for_training:
                include_for_training = True
            else:
                split = "excluded"
                reason = reason or "augmented_parent_not_in_train_or_eval_excluded"
        else:
            if split == "train" and row.usable_for_training:
                include_for_training = True
            if split in {"val", "test"} and row.usable_for_eval:
                include_for_eval = True

        split_rows.append(
            {
                "sample_id": row.sample_id,
                "split_version": SPLIT_VERSION,
                "split": split,
                "canonical_label": row.canonical_label,
                "source_dataset": row.source_dataset,
                "processed_path": row.processed_path,
                "is_augmented": row.is_augmented,
                "parent_sample_key": row.parent_sample_key,
                "dedupe_cluster_id": row.dedupe_cluster_id,
                "label_confidence": row.label_confidence,
                "technical_quality": row.technical_quality,
                "include_for_training": include_for_training,
                "include_for_eval": include_for_eval,
                "exclusion_reason": reason,
            }
        )

    splits = pd.DataFrame(split_rows)
    splits.to_csv(data_lake / "manifests" / "split_manifest_v1.csv", index=False)
    return splits


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown without optional dependencies."""
    if df.empty:
        return "_No rows._"
    string_df = df.astype(str)
    headers = list(string_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in string_df.itertuples(index=False):
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_reports(manifest: pd.DataFrame, splits: pd.DataFrame, dedupe: pd.DataFrame, data_lake: Path) -> None:
    reports_dir = data_lake / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    class_counts = (
        manifest.groupby(["canonical_label", "is_augmented"])
        .size()
        .reset_index(name="count")
        .sort_values(["canonical_label", "is_augmented"])
    )
    class_counts.to_csv(reports_dir / "class_distribution_v1.csv", index=False)

    split_counts = (
        splits.groupby(["split", "canonical_label", "is_augmented"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "canonical_label", "is_augmented"])
    )
    split_counts.to_csv(reports_dir / "split_distribution_v1.csv", index=False)

    quality = manifest[
        [
            "sample_id",
            "canonical_label",
            "is_augmented",
            "duration_sec_processed",
            "rms",
            "peak",
            "clipped_fraction",
            "silence_fraction",
            "technical_quality",
            "exclusion_reason",
        ]
    ]
    quality.to_csv(reports_dir / "quality_report_v1.csv", index=False)

    exact_dup_clusters = int((dedupe["exact_audio_duplicate_count"] > 1).sum())
    eval_eligible = manifest["usable_for_eval"].sum()
    train_included = splits["include_for_training"].sum()
    eval_included = splits["include_for_eval"].sum()

    md = f"""# Phase A Data Audit v1

## Summary

- Source dataset: `{SOURCE_DATASET}`
- Processing version: `{PROCESSING_VERSION}`
- Split version: `{SPLIT_VERSION}`
- Total scanned samples: {len(manifest)}
- Usable original eval candidates: {int(eval_eligible)}
- Training rows included by split manifest: {int(train_included)}
- Validation/test rows included by split manifest: {int(eval_included)}
- Rows flagged as exact audio duplicates: {exact_dup_clusters}

## Class Counts

{markdown_table(class_counts)}

## Split Counts

{markdown_table(split_counts)}

## Important Notes

- Existing `_aug_` files are never allowed into validation/test.
- Existing `_aug_` files are included for training only when their parent original is in the train split.
- No external dataset has been downloaded by this script.
- Current source labels are assigned default label confidence `{LABEL_CONFIDENCE_BY_SOURCE[SOURCE_DATASET]}` because Donate-a-Cry-style labels are parent-reported.
- Infant IDs are parsed only when a UUID-like filename prefix exists; otherwise leakage risk remains high.

## Files Written

- `data_lake/manifests/raw_manifest.csv`
- `data_lake/manifests/processed_manifest.csv`
- `data_lake/manifests/dedupe_clusters.csv`
- `data_lake/manifests/split_manifest_v1.csv`
- `data_lake/reports/class_distribution_v1.csv`
- `data_lake/reports/split_distribution_v1.csv`
- `data_lake/reports/quality_report_v1.csv`
"""
    (reports_dir / "data_audit_v1.md").write_text(md, encoding="utf-8")


def extract_features(manifest: pd.DataFrame, splits: pd.DataFrame, data_lake: Path) -> None:
    feature_dir = data_lake / "features" / "v1_handcrafted_411"
    feature_dir.mkdir(parents=True, exist_ok=True)
    selected = splits[
        (splits["include_for_training"] | splits["include_for_eval"])
        & (splits["processed_path"].astype(str) != "")
    ].copy()
    if selected.empty:
        return

    manifest_by_id = manifest.set_index("sample_id")
    extractor = FeatureExtractor(sr=LEGACY_SAMPLE_RATE)
    rows = []
    features = []
    labels = []

    for split_row in tqdm(selected.itertuples(index=False), total=len(selected), desc="Extracting 411-d features"):
        row = manifest_by_id.loc[split_row.sample_id]
        path = PROJECT_ROOT / row["processed_path"]
        try:
            # Existing handcrafted extractor was designed around 8 kHz.
            audio, _ = librosa.load(path, sr=LEGACY_SAMPLE_RATE, mono=True)
            feat, names = extractor.extract_all(audio)
            features.append(feat.astype(np.float32))
            labels.append(CLASSES.index(split_row.canonical_label))
            rows.append(
                {
                    "sample_id": split_row.sample_id,
                    "split": split_row.split,
                    "canonical_label": split_row.canonical_label,
                    "processed_path": split_row.processed_path,
                }
            )
        except Exception as exc:
            print(f"[WARN] feature extraction failed for {split_row.sample_id}: {exc}")

    if not features:
        return
    np.savez_compressed(
        feature_dir / "features_411.npz",
        X=np.vstack(features).astype(np.float32),
        y=np.array(labels, dtype=np.int64),
        sample_ids=np.array([r["sample_id"] for r in rows]),
    )
    pd.DataFrame(rows).to_csv(feature_dir / "features_411_manifest.csv", index=False)
    if "names" in locals():
        (feature_dir / "feature_names.json").write_text(json.dumps(list(names), indent=2), encoding="utf-8")


def write_dataset_card(data_lake: Path) -> None:
    card = f"""# Dataset Card: Phase A Local v1

This dataset card describes the prepared local data generated by `scripts/phase_a_prepare_data.py`.

## Scope

- Source: existing local `data/raw/<class>/` tree.
- Classes: `{', '.join(CLASSES)}`
- Processing: 16 kHz mono WAV, no destructive edits to raw files.
- Validation/test: originals only.
- Training: originals plus existing augmentations only when parent original is in train.

## Limitations

- Labels are weak parent-reported Donate-a-Cry-style labels.
- No new external datasets are included yet.
- Infant identity is not fully reliable unless UUID-like filenames are present.
- Existing augmented files are useful for training regularization only, not evidence of more real data.

## Next Data Additions

Add downloaded datasets under `data_lake/raw/<dataset_name>/`, then extend the manifest builder to register their provenance, licenses, and label mappings.
"""
    (data_lake / "DATASET_CARD_v1.md").write_text(card, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase A data artifacts.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--data-lake", type=Path, default=PROJECT_ROOT / "data_lake")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_lake.mkdir(parents=True, exist_ok=True)
    files = list(iter_audio_files(args.raw_dir))
    if not files:
        raise SystemExit(f"No audio files found under {args.raw_dir}")

    manifest = prepare_manifest(args.raw_dir, args.data_lake)
    dedupe = write_dedupe_report(manifest, args.data_lake)
    splits = create_splits(manifest, args.data_lake, args.seed)
    write_reports(manifest, splits, dedupe, args.data_lake)
    write_dataset_card(args.data_lake)
    if args.extract_features:
        extract_features(manifest, splits, args.data_lake)

    print("\nPhase A data preparation complete.")
    print(f"Manifest: {args.data_lake / 'manifests' / 'processed_manifest.csv'}")
    print(f"Splits:   {args.data_lake / 'manifests' / 'split_manifest_v1.csv'}")
    print(f"Report:   {args.data_lake / 'reports' / 'data_audit_v1.md'}")


if __name__ == "__main__":
    main()
