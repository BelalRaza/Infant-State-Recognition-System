"""Test-time augmentation feature bank for AST and aux-adapted AST.

For each audio file we extract embeddings from multiple deterministic time
crops of the canonical 10-second window plus the full window, average them in
embedding space, and save as ``ast_tta_embeddings.npz`` and (when an
auxiliary-adapted AST checkpoint is present) ``ast_aux_tta_embeddings.npz``.

These banks plug into ``phase2b_tune_svm.py`` via ``DEFAULT_FEATURE_FILES`` and
are then included in the weighted ensemble automatically. TTA at extraction
time is a low-risk way to squeeze a small but real macro-F1 gain out of fixed
encoders without retraining anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2a.config import (
    AST_MODEL_NAME,
    FEATURES_DIR,
    MAX_DURATION_SEC,
    SAMPLE_RATE,
    SUPERVISED_MANIFEST,
)
from src.phase2a.data import load_audio, load_or_create_splits, resolve_audio_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SUPERVISED_MANIFEST)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--ast-model-name-or-path", type=str, default=AST_MODEL_NAME)
    parser.add_argument(
        "--ast-aux-model-dir",
        type=Path,
        default=PROJECT_ROOT / "results/phase2a/auxiliary_adapted_ast",
        help="Optional adapted AST encoder dir; if present, a TTA bank is built for it too.",
    )
    parser.add_argument("--crop-seconds", type=float, default=7.0)
    parser.add_argument("--num-crops", type=int, default=4)
    parser.add_argument("--include-full", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_crops(audio: np.ndarray, sr: int, crop_seconds: float, num_crops: int, include_full: bool) -> list[np.ndarray]:
    """Return a list of fixed-length crops covering the full audio.

    All crops are zero-padded back to the canonical ``MAX_DURATION_SEC`` length
    so the AST processor can batch them together. This preserves the encoder
    behaviour while giving the model multiple temporal viewpoints.
    """
    target_len = int(sr * MAX_DURATION_SEC)
    crop_len = min(int(sr * crop_seconds), target_len)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    audio = audio[:target_len]

    crops: list[np.ndarray] = []
    if num_crops <= 0:
        crops.append(audio.copy())
    else:
        if num_crops == 1:
            starts = [(target_len - crop_len) // 2]
        else:
            starts = np.linspace(0, target_len - crop_len, num=num_crops, dtype=int).tolist()
        for start in starts:
            chunk = np.zeros(target_len, dtype=audio.dtype)
            window = audio[start : start + crop_len]
            chunk[: len(window)] = window
            crops.append(chunk)
    if include_full:
        crops.append(audio.copy())
    return crops


def extract_tta_bank(
    manifest: pd.DataFrame,
    output_path: Path,
    ast_model_name_or_path: str | Path,
    crop_seconds: float,
    num_crops: int,
    include_full: bool,
    batch_size: int,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"Skipping (already exists): {output_path}")
        return

    from src.phase2a.embeddings import EmbeddingExtractor  # heavy import deferred

    extractor = EmbeddingExtractor("ast", ast_model_name_or_path=ast_model_name_or_path)

    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    labels = manifest["label_idx"].to_numpy(dtype=np.int64)
    splits = manifest["split"].astype(str).to_numpy()

    pending_audios: list[np.ndarray] = []
    pending_owner: list[int] = []
    accum: dict[int, list[np.ndarray]] = {}

    def flush() -> None:
        if not pending_audios:
            return
        embs = extractor.extract_batch(pending_audios)
        for owner_idx, emb in zip(pending_owner, embs):
            accum.setdefault(owner_idx, []).append(emb)
        pending_audios.clear()
        pending_owner.clear()

    progress = tqdm(manifest.itertuples(index=False), total=len(manifest), desc=f"TTA AST -> {output_path.name}")
    for idx, row in enumerate(progress):
        audio_path = resolve_audio_path(row.processed_path, PROJECT_ROOT)
        audio = load_audio(audio_path)
        crops = build_crops(audio, SAMPLE_RATE, crop_seconds, num_crops, include_full)
        for crop in crops:
            pending_audios.append(crop)
            pending_owner.append(idx)
            if len(pending_audios) >= batch_size:
                flush()
    flush()

    feature_dim = next(iter(accum.values()))[0].shape[0]
    X = np.zeros((len(manifest), feature_dim), dtype=np.float32)
    for owner_idx, emb_list in accum.items():
        X[owner_idx] = np.mean(np.stack(emb_list, axis=0), axis=0).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=X, y=labels, sample_ids=sample_ids, splits=splits)
    print(f"Saved TTA feature bank: {output_path} (shape={X.shape})")


def main() -> None:
    args = parse_args()
    splits_df = load_or_create_splits(args.manifest)

    base_out = args.features_dir / "ast_tta_embeddings.npz"
    extract_tta_bank(
        splits_df,
        base_out,
        ast_model_name_or_path=args.ast_model_name_or_path,
        crop_seconds=args.crop_seconds,
        num_crops=args.num_crops,
        include_full=args.include_full,
        batch_size=args.batch_size,
        force=args.force,
    )

    aux_dir = args.ast_aux_model_dir
    if aux_dir.exists():
        aux_out = args.features_dir / "ast_aux_tta_embeddings.npz"
        extract_tta_bank(
            splits_df,
            aux_out,
            ast_model_name_or_path=aux_dir,
            crop_seconds=args.crop_seconds,
            num_crops=args.num_crops,
            include_full=args.include_full,
            batch_size=args.batch_size,
            force=args.force,
        )
    else:
        print(f"No aux-adapted AST found at {aux_dir}; skipping aux TTA bank.")

    metadata = {
        "crop_seconds": args.crop_seconds,
        "num_crops": args.num_crops,
        "include_full": args.include_full,
        "batch_size": args.batch_size,
        "ast_model_name_or_path": str(args.ast_model_name_or_path),
        "ast_aux_model_dir": str(aux_dir),
    }
    (args.features_dir / "tta_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
