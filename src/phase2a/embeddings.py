"""Pretrained embedding and handcrafted feature extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from tqdm import tqdm

from src.phase2a.config import (
    AST_MODEL_NAME,
    AST_MODEL_NAME_OR_PATH,
    FEATURE_CHECKPOINT_EVERY,
    FEATURES_DIR,
    HF_CACHE_DIR,
    ROOT_DIR,
    SAMPLE_RATE,
    WHISPER_MODEL_NAME,
)
from src.phase2a.data import load_audio, resolve_audio_path


def summarise_frames(matrix: np.ndarray) -> np.ndarray:
    funcs = [
        np.mean,
        np.std,
        np.min,
        np.max,
        lambda row: float(sp_stats.skew(row)),
        lambda row: float(sp_stats.kurtosis(row)),
    ]
    summary: list[float] = []
    for row in matrix:
        clean = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        summary.extend(float(f(clean)) for f in funcs)
    return np.asarray(summary, dtype=np.float32)


def extract_handcrafted_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract compact handcrafted features for the classical ensemble branch."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=1024, hop_length=320, n_mels=64)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=320)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=320)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=320)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=320)
    rms = librosa.feature.rms(y=audio, hop_length=320)
    spectral = np.vstack([zcr, centroid, bandwidth, rolloff, rms])
    return np.concatenate(
        [
            summarise_frames(mfcc),
            summarise_frames(delta1),
            summarise_frames(delta2),
            summarise_frames(spectral),
        ]
    ).astype(np.float32)


class EmbeddingExtractor:
    """Thin wrapper around HuggingFace audio models."""

    def __init__(self, model_key: str, device: str | None = None, ast_model_name_or_path: str | Path | None = None):
        self.model_key = model_key
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
        if model_key == "ast":
            from transformers import ASTFeatureExtractor, AutoModel

            ast_source = str(ast_model_name_or_path or AST_MODEL_NAME_OR_PATH)
            self.processor = ASTFeatureExtractor.from_pretrained(ast_source, cache_dir=HF_CACHE_DIR)
            self.model = AutoModel.from_pretrained(ast_source, cache_dir=HF_CACHE_DIR).to(self.device)
        elif model_key == "whisper":
            from transformers import WhisperFeatureExtractor, WhisperModel

            self.processor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_NAME, cache_dir=HF_CACHE_DIR)
            self.model = WhisperModel.from_pretrained(WHISPER_MODEL_NAME, cache_dir=HF_CACHE_DIR).encoder.to(self.device)
        else:
            raise ValueError(f"Unsupported embedding model: {model_key}")
        self.model.eval()

    @torch.no_grad()
    def extract_batch(self, audios: list[np.ndarray]) -> np.ndarray:
        if not audios:
            return np.empty((0, 0), dtype=np.float32)
        if self.model_key == "ast":
            inputs = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
        elif self.model_key == "whisper":
            inputs = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            features = inputs.input_features.to(self.device)
            outputs = self.model(input_features=features)
            emb = outputs.last_hidden_state.mean(dim=1)
        else:
            raise ValueError(self.model_key)
        return emb.detach().cpu().numpy().astype(np.float32)

    def extract_one(self, audio: np.ndarray) -> np.ndarray:
        return self.extract_batch([audio])[0]


def extract_feature_bank(
    manifest: pd.DataFrame,
    model_keys: list[str],
    output_dir: Path = FEATURES_DIR,
    project_root: Path = ROOT_DIR,
    force: bool = False,
    checkpoint_every: int = FEATURE_CHECKPOINT_EVERY,
    ast_model_name_or_path: str | Path | None = None,
    batch_size: int = 8,
) -> dict[str, Path]:
    """Extract requested pretrained embeddings and handcrafted features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    labels = manifest["label_idx"].to_numpy(dtype=np.int64)
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    splits = manifest["split"].astype(str).to_numpy()

    for key in model_keys:
        output_key = key
        if key == "ast" and ast_model_name_or_path and str(ast_model_name_or_path) != AST_MODEL_NAME:
            output_key = "ast_aux_adapted"
        out_path = output_dir / f"{output_key}_embeddings.npz"
        if out_path.exists() and not force:
            paths[output_key] = out_path
            continue
        extractor = EmbeddingExtractor(key, ast_model_name_or_path=ast_model_name_or_path)
        partial_path = output_dir / f"{output_key}_embeddings.partial.npz"
        vectors = _load_partial(partial_path, force=force)
        pending_audios: list[np.ndarray] = []
        pending_ids: list[str] = []
        since_checkpoint = 0
        progress = tqdm(manifest.itertuples(index=False), total=len(manifest), desc=f"Extracting {key}")
        for row in progress:
            sample_id = str(row.sample_id)
            if sample_id in vectors:
                continue
            audio_path = resolve_audio_path(row.processed_path, project_root)
            pending_audios.append(load_audio(audio_path))
            pending_ids.append(sample_id)
            if len(pending_audios) >= batch_size:
                embeddings = extractor.extract_batch(pending_audios)
                for sid, emb in zip(pending_ids, embeddings):
                    vectors[sid] = emb
                since_checkpoint += len(pending_ids)
                pending_audios.clear()
                pending_ids.clear()
                if checkpoint_every > 0 and since_checkpoint >= checkpoint_every:
                    _save_partial(partial_path, vectors)
                    since_checkpoint = 0
        if pending_audios:
            embeddings = extractor.extract_batch(pending_audios)
            for sid, emb in zip(pending_ids, embeddings):
                vectors[sid] = emb
        _save_partial(partial_path, vectors)
        X = np.vstack([vectors[str(sample_id)] for sample_id in sample_ids]).astype(np.float32)
        np.savez_compressed(out_path, X=X, y=labels, sample_ids=sample_ids, splits=splits)
        partial_path.unlink(missing_ok=True)
        paths[output_key] = out_path

    handcrafted_path = output_dir / "handcrafted_features.npz"
    if not handcrafted_path.exists() or force:
        partial_path = output_dir / "handcrafted_features.partial.npz"
        feats = _load_partial(partial_path, force=force)
        for idx, row in enumerate(tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Extracting handcrafted"), start=1):
            sample_id = str(row.sample_id)
            if sample_id in feats:
                continue
            audio_path = resolve_audio_path(row.processed_path, project_root)
            audio = load_audio(audio_path)
            feats[sample_id] = extract_handcrafted_features(audio)
            if checkpoint_every > 0 and idx % checkpoint_every == 0:
                _save_partial(partial_path, feats)
        _save_partial(partial_path, feats)
        Xh = np.vstack([feats[str(sample_id)] for sample_id in sample_ids]).astype(np.float32)
        np.savez_compressed(handcrafted_path, X=Xh, y=labels, sample_ids=sample_ids, splits=splits)
        partial_path.unlink(missing_ok=True)
    paths["handcrafted"] = handcrafted_path

    metadata = {
        "rows": int(len(manifest)),
        "models": model_keys,
        "sample_rate": SAMPLE_RATE,
        "hf_cache_dir": str(HF_CACHE_DIR),
        "checkpoint_every": checkpoint_every,
        "ast_model_name_or_path": str(ast_model_name_or_path or AST_MODEL_NAME),
        "batch_size": batch_size,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    (output_dir / "feature_bank_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def _load_partial(path: Path, force: bool = False) -> dict[str, np.ndarray]:
    if force or not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    sample_ids = data["sample_ids"].astype(str)
    X = data["X"]
    return {sample_id: X[idx] for idx, sample_id in enumerate(sample_ids)}


def _save_partial(path: Path, vectors: dict[str, np.ndarray]) -> None:
    if not vectors:
        return
    sample_ids = np.asarray(list(vectors.keys()), dtype=object)
    X = np.vstack([vectors[sample_id] for sample_id in sample_ids]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, sample_ids=sample_ids)

