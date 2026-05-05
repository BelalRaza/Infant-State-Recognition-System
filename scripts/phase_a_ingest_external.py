#!/usr/bin/env python3
"""Ingest external infant-cry datasets into the Phase A data lake.

The script is conservative by design:
- converts external audio to canonical 16 kHz mono WAV,
- maps only compatible labels into the 5-class cause task,
- keeps detection/auxiliary labels separate,
- flags known synthetic augmentations,
- deduplicates against the local Phase A manifest,
- keeps validation/test as local originals only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSES  # noqa: E402


CANONICAL_SAMPLE_RATE = 16_000
PROCESSING_VERSION = "v2_external_16k_mono"
UNIFIED_SPLIT_VERSION = "split_v2_local_eval_external_train"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".3gp", ".aac"}


DATASET_TRUST = {
    "baby_cry_pattern_archive": 0.45,
    "baby_crying_sounds": 0.45,
    "baby_cry_sense": 0.50,
    "uac_butembo": 0.35,
}


LABEL_ALIASES = {
    "hunger": "hunger",
    "hungry": "hunger",
    "faim": "hunger",
    "belly_pain": "belly_pain",
    "belly pain": "belly_pain",
    "bellypain": "belly_pain",
    "pain": "belly_pain",
    "stomachache": "belly_pain",
    "colic": "belly_pain",
    "burping": "burping",
    "needs burping": "burping",
    "burp": "burping",
    "rot": "burping",
    "discomfort": "discomfort",
    "uncomfortable": "discomfort",
    "inconfort": "discomfort",
    "cold_hot": "discomfort",
    "cold hot": "discomfort",
    "cold": "discomfort",
    "hot": "discomfort",
    "cold/hot": "discomfort",
    "scared": "discomfort",
    "lonely": "discomfort",
    "uncomfortable": "discomfort",
    "diaper": "discomfort",
    "tired": "tiredness",
    "tiredness": "tiredness",
    "sleepy": "tiredness",
    "sommeil": "tiredness",
    "fatigue": "tiredness",
}


AUX_LABELS = {
    "silence": "non_cry_silence",
    "noise": "non_cry_noise",
    "laugh": "non_cry_laugh",
    "rire": "non_cry_laugh",
    "pleurs": "cry_unlabeled",
    "cry": "cry_unlabeled",
    "no cry": "non_cry",
    "no_cry": "non_cry",
    "quiet": "non_cry_quiet",
    "environment": "non_cry_environment",
    "don't know": "unknown_or_uncertain",
    "dont know": "unknown_or_uncertain",
    "autres": "other_cry_or_unknown",
    "other": "other_cry_or_unknown",
    "others": "other_cry_or_unknown",
}


AUGMENTATION_MARKERS = (
    "_aug_",
    "_pitch_shifted",
    "_time_stretched",
    "_volume_changed",
    "_with_noise",
    "_noise",
    "_stretched",
    "_shifted",
)


@dataclass
class ExternalRecord:
    sample_id: str
    source_dataset: str
    source_file: str
    raw_path: str
    processed_path: str
    raw_sha256: str
    audio_sha256_16k: str
    audio_fingerprint: str
    source_label: str
    canonical_label: str
    task_tags: str
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
    usable_for_cause5_training: bool
    usable_for_aux_pretraining: bool
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
    return hashlib.sha256(np.asarray(audio, dtype=np.float32).tobytes()).hexdigest()


def audio_fingerprint(audio: np.ndarray, sr: int) -> str:
    """Stable content fingerprint for copied/re-encoded near-duplicates."""
    if len(audio) == 0:
        return ""
    target_len = min(len(audio), sr * 8)
    clip = audio[:target_len].astype(np.float32)
    peak = float(np.max(np.abs(clip)))
    if peak > 0:
        clip = clip / peak
    mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=32, n_fft=512, hop_length=256)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    resized = librosa.util.fix_length(log_mel, size=64, axis=1)
    centered = resized - float(np.mean(resized))
    bits = (centered > 0).astype(np.uint8).reshape(-1)
    packed = np.packbits(bits)
    duration_bucket = round(len(audio) / sr, 1)
    return f"{duration_bucket:.1f}:{hashlib.sha256(packed.tobytes()).hexdigest()[:24]}"


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "sample"


def normalized_label(value: str) -> str:
    value = value.replace("-", "_").replace(".", "_").strip().lower()
    value = re.sub(r"\s+", " ", value.replace("_", " "))
    return value


def infer_source_label(dataset_name: str, dataset_root: Path, path: Path) -> str:
    if dataset_name.startswith("Nexdata__Infant_Cry_Speech_Data_by_Mobile_Phone"):
        return "cry"
    parts = path.relative_to(dataset_root).parts[:-1]
    for part in reversed(parts):
        norm = normalized_label(part)
        if norm in LABEL_ALIASES or norm in AUX_LABELS:
            return norm
    return normalized_label(path.parent.name)


def map_label(source_label: str) -> tuple[str, str, float, bool, bool]:
    norm = normalized_label(source_label)
    if norm in LABEL_ALIASES:
        canonical = LABEL_ALIASES[norm]
        weak_mapped = norm not in {
            "hunger",
            "hungry",
            "belly pain",
            "belly_pain",
            "burping",
            "discomfort",
            "tired",
            "tiredness",
        }
        task = "cause_5class_weak_mapped" if weak_mapped else "cause_5class"
        label_penalty = 0.15 if weak_mapped else 0.0
        return canonical, task, label_penalty, True, True
    if norm in AUX_LABELS:
        return AUX_LABELS[norm], "auxiliary_detection_or_ssl", 0.10, False, True
    return norm or "unknown", "unknown_unmapped", 0.30, False, False


def apply_dataset_policy(
    dataset_name: str,
    source_label: str,
    canonical_label: str,
    task_tags: str,
    usable_cause5: bool,
    usable_aux: bool,
) -> tuple[str, str, bool, bool]:
    if dataset_name == "uac_butembo" and canonical_label == "other_cry_or_unknown":
        return canonical_label, "quarantine_audit_only", False, False
    if dataset_name == "uac_butembo" and canonical_label == "cry_unlabeled":
        return canonical_label, "auxiliary_weak_cry_ssl", False, True
    return canonical_label, task_tags, usable_cause5, usable_aux


def parse_augmentation(path: Path) -> tuple[bool, str]:
    stem = path.stem.lower()
    for marker in AUGMENTATION_MARKERS:
        if marker in stem:
            return True, marker.strip("_")
    return False, ""


def load_audio(path: Path) -> tuple[np.ndarray, int, float]:
    try:
        info = sf.info(str(path))
        raw_sr = int(info.samplerate)
        raw_duration = float(info.frames / info.samplerate) if info.samplerate else 0.0
    except Exception:
        raw_sr = 0
        raw_duration = 0.0
    audio, _ = librosa.load(path, sr=CANONICAL_SAMPLE_RATE, mono=True)
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
    if len(audio) / sr < 0.50:
        score -= 0.35
    if rms < 0.003:
        score -= 0.25
    if peak < 0.015:
        score -= 0.20
    if clipped_fraction > 0.02:
        score -= 0.20
    if silence_fraction > 0.90:
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


def iter_external_audio(raw_roots: list[Path]):
    for raw_root in raw_roots:
        if not raw_root.exists():
            continue
        yield from iter_external_audio_root(raw_root)


def iter_external_audio_root(raw_root: Path):
    for dataset_root in sorted(raw_root.iterdir()):
        if not dataset_root.is_dir():
            continue
        for path in sorted(dataset_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                yield dataset_root.name, dataset_root, path


def ingest_external(raw_roots: list[Path], data_lake: Path) -> pd.DataFrame:
    processed_root = data_lake / "processed" / PROCESSING_VERSION
    processed_root.mkdir(parents=True, exist_ok=True)
    records: list[ExternalRecord] = []
    counters: Counter[str] = Counter()
    audio_files = list(iter_external_audio(raw_roots))

    for dataset_name, dataset_root, raw_path in tqdm(audio_files, desc="Ingesting external audio"):
        source_label = infer_source_label(dataset_name, dataset_root, raw_path)
        canonical_label, task_tags, label_penalty, usable_cause5, usable_aux = map_label(source_label)
        canonical_label, task_tags, usable_cause5, usable_aux = apply_dataset_policy(
            dataset_name, source_label, canonical_label, task_tags, usable_cause5, usable_aux
        )
        is_augmented, aug_tag = parse_augmentation(raw_path)
        counters[dataset_name] += 1
        sample_id = f"{dataset_name}_{counters[dataset_name]:07d}"
        out_dir = processed_root / dataset_name / safe_name(canonical_label)
        out_dir.mkdir(parents=True, exist_ok=True)
        processed_path = out_dir / f"{sample_id}.wav"
        raw_sha = sha256_file(raw_path)
        load_error = ""

        try:
            audio, raw_sr, raw_duration = load_audio(raw_path)
            sf.write(str(processed_path), audio, CANONICAL_SAMPLE_RATE)
            duration_processed = len(audio) / CANONICAL_SAMPLE_RATE
            audio_sha = sha256_audio(audio)
            fingerprint = audio_fingerprint(audio, CANONICAL_SAMPLE_RATE)
            metrics = compute_quality(audio, CANONICAL_SAMPLE_RATE)
        except Exception as exc:
            load_error = str(exc)
            raw_sr = 0
            raw_duration = 0.0
            duration_processed = 0.0
            audio_sha = ""
            fingerprint = ""
            metrics = compute_quality(np.array([], dtype=np.float32), CANONICAL_SAMPLE_RATE)
            processed_path = Path("")

        exclusion_reason = ""
        if load_error:
            exclusion_reason = f"load_error: {load_error}"
        elif metrics["technical_quality"] < 0.45:
            exclusion_reason = "technical_quality_below_0.45"
        elif task_tags == "unknown_unmapped":
            exclusion_reason = "unmapped_label"
        elif task_tags == "quarantine_audit_only":
            exclusion_reason = "quarantined_uac_autres_not_infant_cause_or_clean_aux"

        dataset_conf = DATASET_TRUST.get(dataset_name, 0.35)
        label_confidence = max(0.05, dataset_conf - label_penalty - (0.05 if is_augmented else 0.0))
        usable_for_cause5_training = usable_cause5 and not exclusion_reason
        usable_for_aux_pretraining = usable_aux and not load_error and metrics["technical_quality"] >= 0.35

        records.append(
            ExternalRecord(
                sample_id=sample_id,
                source_dataset=dataset_name,
                source_file=raw_path.name,
                raw_path=str(raw_path.relative_to(PROJECT_ROOT)),
                processed_path=str(processed_path.relative_to(PROJECT_ROOT)) if processed_path else "",
                raw_sha256=raw_sha,
                audio_sha256_16k=audio_sha,
                audio_fingerprint=fingerprint,
                source_label=source_label,
                canonical_label=canonical_label,
                task_tags=task_tags,
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
                label_confidence=label_confidence,
                usable_for_cause5_training=usable_for_cause5_training,
                usable_for_aux_pretraining=usable_for_aux_pretraining,
                usable_for_eval=False,
                exclusion_reason=exclusion_reason,
                processing_version=PROCESSING_VERSION,
            )
        )

    external = pd.DataFrame([asdict(record) for record in records])
    manifest_dir = data_lake / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    external.to_csv(manifest_dir / "external_manifest_v2.csv", index=False)
    return external


def build_unified_manifests(external: pd.DataFrame, data_lake: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_manifest_path = data_lake / "manifests" / "processed_manifest.csv"
    local_split_path = data_lake / "manifests" / "split_manifest_v1.csv"
    if not local_manifest_path.exists() or not local_split_path.exists():
        raise RuntimeError("Run scripts/phase_a_prepare_data.py before external ingestion.")

    local_manifest = pd.read_csv(local_manifest_path)
    local_splits = pd.read_csv(local_split_path)
    local_split_by_id = local_splits.set_index("sample_id")

    local_rows = []
    for row in local_manifest.itertuples(index=False):
        split_row = local_split_by_id.loc[row.sample_id] if row.sample_id in local_split_by_id.index else None
        split = split_row["split"] if split_row is not None else "excluded"
        include_training = bool(split_row["include_for_training"]) if split_row is not None else False
        include_eval = bool(split_row["include_for_eval"]) if split_row is not None else False
        local_fingerprint = ""
        processed_path = PROJECT_ROOT / row.processed_path
        if processed_path.exists():
            try:
                audio, _ = librosa.load(processed_path, sr=CANONICAL_SAMPLE_RATE, mono=True)
                local_fingerprint = audio_fingerprint(audio.astype(np.float32), CANONICAL_SAMPLE_RATE)
            except Exception:
                local_fingerprint = ""
        local_rows.append(
            {
                "sample_id": row.sample_id,
                "source_dataset": row.source_dataset,
                "processed_path": row.processed_path,
                "audio_sha256_16k": row.audio_sha256_16k,
                "audio_fingerprint": local_fingerprint,
                "canonical_label": row.canonical_label,
                "task_tags": row.task_tags,
                "is_augmented": bool(row.is_augmented),
                "label_confidence": row.label_confidence,
                "technical_quality": row.technical_quality,
                "split": split,
                "include_for_cause5_training": include_training,
                "include_for_aux_pretraining": False,
                "include_for_eval": include_eval,
                "dedupe_status": "kept",
                "exclusion_reason": row.exclusion_reason if isinstance(row.exclusion_reason, str) else "",
            }
        )

    unified = pd.DataFrame(local_rows)
    existing_hashes = set(unified["audio_sha256_16k"].dropna().astype(str))
    existing_fingerprints = set(unified["audio_fingerprint"].dropna().astype(str))
    existing_fingerprints.discard("")
    external_rows = []
    seen_external_hashes: set[str] = set()
    seen_external_fingerprints: set[str] = set()

    for row in external.itertuples(index=False):
        audio_hash = str(row.audio_sha256_16k)
        fingerprint = str(row.audio_fingerprint)
        duplicate_hash = bool(audio_hash and (audio_hash in existing_hashes or audio_hash in seen_external_hashes))
        duplicate_fingerprint = bool(
            fingerprint and (fingerprint in existing_fingerprints or fingerprint in seen_external_fingerprints)
        )
        duplicate = duplicate_hash or duplicate_fingerprint
        if audio_hash:
            seen_external_hashes.add(audio_hash)
        if fingerprint:
            seen_external_fingerprints.add(fingerprint)
        cause_label = row.canonical_label in CLASSES
        authentic_original = not bool(row.is_augmented)
        include_cause = bool(row.usable_for_cause5_training and cause_label and authentic_original and not duplicate)
        include_aux = bool(row.usable_for_aux_pretraining and not duplicate)
        exclusion = row.exclusion_reason if isinstance(row.exclusion_reason, str) else ""
        if duplicate:
            if duplicate_hash:
                exclusion = exclusion or "duplicate_audio_hash"
            if duplicate_fingerprint:
                exclusion = exclusion or "near_duplicate_audio_fingerprint"
        if cause_label and row.usable_for_cause5_training and not authentic_original:
            exclusion = exclusion or "external_augmented_not_in_authentic_base"
        external_rows.append(
            {
                "sample_id": row.sample_id,
                "source_dataset": row.source_dataset,
                "processed_path": row.processed_path,
                "audio_sha256_16k": row.audio_sha256_16k,
                "audio_fingerprint": row.audio_fingerprint,
                "canonical_label": row.canonical_label,
                "task_tags": row.task_tags,
                "is_augmented": bool(row.is_augmented),
                "label_confidence": row.label_confidence,
                "technical_quality": row.technical_quality,
                "split": "external_train" if include_cause else ("aux_pretrain" if include_aux else "excluded"),
                "include_for_cause5_training": include_cause,
                "include_for_aux_pretraining": include_aux,
                "include_for_eval": False,
                "dedupe_status": "duplicate" if duplicate else "kept",
                "exclusion_reason": exclusion,
            }
        )

    if external_rows:
        unified = pd.concat([unified, pd.DataFrame(external_rows)], ignore_index=True)

    unified.to_csv(data_lake / "manifests" / "unified_manifest_v2.csv", index=False)

    cause_train = unified[
        (unified["include_for_cause5_training"])
        & (unified["canonical_label"].isin(CLASSES))
        & (~unified["is_augmented"])
        & (unified["processed_path"].astype(str) != "")
    ].copy()
    cause_train.to_csv(data_lake / "manifests" / "cause5_train_manifest_v2.csv", index=False)
    cause_train.to_csv(data_lake / "manifests" / "cause5_authentic_manifest_v2.csv", index=False)
    dedupe_report = unified[unified["dedupe_status"] == "duplicate"].copy()
    dedupe_report.to_csv(data_lake / "manifests" / "near_duplicate_report_v2.csv", index=False)
    return unified, cause_train


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    df = df.astype(str)
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_reports(external: pd.DataFrame, unified: pd.DataFrame, cause_train: pd.DataFrame, data_lake: Path) -> None:
    reports_dir = data_lake / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    external_counts = (
        external.groupby(["source_dataset", "canonical_label", "task_tags", "is_augmented"])
        .size()
        .reset_index(name="count")
        .sort_values(["source_dataset", "canonical_label", "is_augmented"])
    )
    train_counts = (
        cause_train.groupby(["source_dataset", "canonical_label", "is_augmented"])
        .size()
        .reset_index(name="count")
        .sort_values(["source_dataset", "canonical_label", "is_augmented"])
    )
    unified_counts = (
        unified.groupby(["split", "canonical_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "canonical_label"])
    )
    external_counts.to_csv(reports_dir / "external_distribution_v2.csv", index=False)
    train_counts.to_csv(reports_dir / "cause5_train_distribution_v2.csv", index=False)
    unified_counts.to_csv(reports_dir / "unified_split_distribution_v2.csv", index=False)

    summary = {
        "external_rows": int(len(external)),
        "unified_rows": int(len(unified)),
        "cause5_authentic_rows": int(len(cause_train)),
        "external_duplicate_rows": int((unified["dedupe_status"] == "duplicate").sum()),
        "external_unmapped_rows": int((external["task_tags"] == "unknown_unmapped").sum()),
        "quarantined_uac_autres_rows": int((external["task_tags"] == "quarantine_audit_only").sum()),
        "aux_pretraining_rows": int(unified["include_for_aux_pretraining"].sum()),
    }
    (reports_dir / "phase_a_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = f"""# Phase A External Data Report v2

## Summary

- External rows ingested: {summary["external_rows"]}
- Unified manifest rows: {summary["unified_rows"]}
- Authentic cause-5 rows: {summary["cause5_authentic_rows"]}
- Rows marked duplicate by exact 16 kHz hash or content fingerprint: {summary["external_duplicate_rows"]}
- Auxiliary pretraining rows: {summary["aux_pretraining_rows"]}
- UAC `autres` rows quarantined/audit-only: {summary["quarantined_uac_autres_rows"]}
- Validation/test policy: local originals only; external data is not used for evaluation.
- Cause-5 policy: authentic originals only; no pre-existing augmented clips in the final base manifest.

## External Distribution

{markdown_table(external_counts)}

## Cause-5 Training Distribution

{markdown_table(train_counts)}

## Unified Split Distribution

{markdown_table(unified_counts)}

## Files Written

- `data_lake/manifests/external_manifest_v2.csv`
- `data_lake/manifests/unified_manifest_v2.csv`
- `data_lake/manifests/cause5_train_manifest_v2.csv`
- `data_lake/manifests/cause5_authentic_manifest_v2.csv`
- `data_lake/manifests/near_duplicate_report_v2.csv`
- `data_lake/reports/external_distribution_v2.csv`
- `data_lake/reports/cause5_train_distribution_v2.csv`
- `data_lake/reports/unified_split_distribution_v2.csv`
- `data_lake/reports/phase_a_v2_summary.json`
"""
    (reports_dir / "external_data_report_v2.md").write_text(md, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest external Phase A datasets.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        action="append",
        default=None,
        help="External raw root. Can be passed multiple times.",
    )
    parser.add_argument("--data-lake", type=Path, default=PROJECT_ROOT / "data_lake")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_roots = args.raw_root or [
        PROJECT_ROOT / "data_lake" / "raw" / "kaggle",
        PROJECT_ROOT / "data_lake" / "raw" / "huggingface",
    ]
    external = ingest_external(raw_roots, args.data_lake)
    unified, cause_train = build_unified_manifests(external, args.data_lake)
    write_reports(external, unified, cause_train, args.data_lake)
    print("External Phase A ingestion complete.")
    print(f"External manifest: {args.data_lake / 'manifests' / 'external_manifest_v2.csv'}")
    print(f"Unified manifest:  {args.data_lake / 'manifests' / 'unified_manifest_v2.csv'}")
    print(f"Cause-5 train:     {args.data_lake / 'manifests' / 'cause5_train_manifest_v2.csv'}")


if __name__ == "__main__":
    main()
