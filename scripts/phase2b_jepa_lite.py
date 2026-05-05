"""Train AudioJEPA-lite and export infant-cry representations.

This is a task-specific AudioJEPA implementation, not a generic contrastive
baseline. It follows the JEPA principle:

    context spectrogram with masked target block
        -> online encoder
        -> predictor
        -> predicted latent target tokens

    clean/full spectrogram
        -> EMA target encoder
        -> target latent tokens

    loss = latent prediction error on masked target tokens only

The learned encoder is then frozen and exported as `jepa_lite_embeddings.npz`
so Phase 2B can treat AudioJEPA as one additional feature view.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    if "-h" in sys.argv or "--help" in sys.argv:
        class _DummyNN:
            Module = object

        torch = None
        nn = _DummyNN()
        F = None
        DataLoader = None
        Dataset = object
    else:
        raise

from src.phase2a.config import FEATURES_DIR, SAMPLE_RATE, SUPERVISED_MANIFEST, UNIFIED_MANIFEST
from src.phase2a.data import load_or_create_splits, resolve_audio_path
from src.phase2b.config import ARTIFACTS_DIR, METRICS_DIR, ROOT_DIR


@dataclass
class AudioJEPAConfig:
    sample_rate: int = SAMPLE_RATE
    duration_sec: float = 4.0
    n_mels: int = 96
    n_fft: int = 1024
    hop_length: int = 320
    fmin: int = 50
    fmax: int = 8000
    patch_size: int = 16
    patch_stride: int = 8
    embed_dim: int = 192
    encoder_depth: int = 4
    encoder_heads: int = 4
    predictor_depth: int = 2
    predictor_heads: int = 4
    predictor_mlp_ratio: float = 2.0
    target_time_ratio: float = 0.30
    target_freq_ratio: float = 0.55
    ema_start: float = 0.99
    ema_end: float = 0.999
    variance_floor: float = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--project-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = AudioJEPAConfig(duration_sec=args.duration_sec)
    out_path = args.output_dir / "jepa_lite_embeddings.npz"
    if out_path.exists() and not args.force:
        print(f"exists: {out_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = build_ssl_manifest(args.project_root)
    dataset = AudioJEPADataset(train_manifest, args.project_root, cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = AudioJEPA(cfg).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.05)
    history = []

    for epoch in range(1, args.epochs + 1):
        momentum = cosine_schedule(epoch - 1, args.epochs, cfg.ema_start, cfg.ema_end)
        row = train_one_epoch(model, loader, optimizer, device, momentum, epoch)
        history.append(row)
        torch.save(
            {
                "model_state": model.online_encoder.state_dict(),
                "target_state": model.target_encoder.state_dict(),
                "predictor_state": model.predictor.state_dict(),
                "config": asdict(cfg),
                "args": vars(args),
                "history": history,
            },
            ARTIFACTS_DIR / "phase2b_audiojepa_last.pt",
        )
        print(row)

    strict = load_or_create_splits(SUPERVISED_MANIFEST)
    embeddings = extract_embeddings(model.online_encoder, strict, args.project_root, cfg, device)
    np.savez_compressed(
        out_path,
        X=embeddings,
        y=strict["label_idx"].to_numpy(dtype=np.int64),
        sample_ids=strict["sample_id"].astype(str).to_numpy(),
        splits=strict["split"].astype(str).to_numpy(),
        config=np.asarray(json.dumps(asdict(cfg)), dtype=object),
    )
    (METRICS_DIR / "phase2b_audiojepa_training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"wrote: {out_path} shape={embeddings.shape}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ssl_manifest(project_root: Path) -> pd.DataFrame:
    strict = load_or_create_splits(SUPERVISED_MANIFEST)
    strict_train = strict[strict["split"] == "train"][["sample_id", "processed_path"]].copy()
    unified = pd.read_csv(UNIFIED_MANIFEST)
    aux = unified[
        (unified["include_for_aux_pretraining"] == True)
        & (unified["dedupe_status"] == "kept")
        & (unified["include_for_eval"] == False)
        & (unified["canonical_label"] != "unknown_or_uncertain")
    ][["sample_id", "processed_path"]].copy()
    out = pd.concat([strict_train, aux], ignore_index=True).drop_duplicates("processed_path")
    out = out[out["processed_path"].map(lambda p: resolve_audio_path(p, project_root).exists())]
    return out.reset_index(drop=True)


class AudioJEPADataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, project_root: Path, cfg: AudioJEPAConfig):
        self.manifest = manifest.reset_index(drop=True)
        self.project_root = project_root
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        path = resolve_audio_path(self.manifest.iloc[idx]["processed_path"], self.project_root)
        y, _ = librosa.load(path, sr=self.cfg.sample_rate, mono=True)
        mel = make_logmel(y, self.cfg, random_crop=True, augment=True)
        context, target_mask = make_context_and_mask(mel, self.cfg)
        return torch.from_numpy(context[None]), torch.from_numpy(mel[None]), torch.from_numpy(target_mask)


def make_logmel(audio: np.ndarray, cfg: AudioJEPAConfig, random_crop: bool, augment: bool) -> np.ndarray:
    target = int(cfg.sample_rate * cfg.duration_sec)
    if len(audio) > target:
        if random_crop:
            start = np.random.randint(0, len(audio) - target + 1)
        else:
            start = (len(audio) - target) // 2
        audio = audio[start : start + target]
    elif len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))

    if augment:
        audio = audio * np.random.uniform(0.90, 1.10)
        audio = audio + np.random.normal(0.0, 0.002, size=len(audio)).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return ((logmel - logmel.mean()) / (logmel.std() + 1e-6)).astype(np.float32)


def make_context_and_mask(mel: np.ndarray, cfg: AudioJEPAConfig) -> tuple[np.ndarray, np.ndarray]:
    context = mel.copy()
    patch_h = cfg.patch_size
    patch_w = cfg.patch_size
    stride = cfg.patch_stride
    grid_h = (mel.shape[0] - patch_h) // stride + 1
    grid_w = (mel.shape[1] - patch_w) // stride + 1

    target_h = max(2, int(grid_h * cfg.target_freq_ratio))
    target_w = max(3, int(grid_w * cfg.target_time_ratio))
    start_h = np.random.randint(0, max(1, grid_h - target_h + 1))
    # Bias toward later time blocks so the model learns cry evolution.
    low_t = max(0, int(grid_w * 0.35))
    high_t = max(low_t + 1, grid_w - target_w + 1)
    start_w = np.random.randint(low_t, high_t)

    mask = np.zeros((grid_h, grid_w), dtype=bool)
    mask[start_h : start_h + target_h, start_w : start_w + target_w] = True

    f0 = start_h * stride
    f1 = min(mel.shape[0], f0 + patch_h + (target_h - 1) * stride)
    t0 = start_w * stride
    t1 = min(mel.shape[1], t0 + patch_w + (target_w - 1) * stride)
    context[f0:f1, t0:t1] = 0.0

    # Add a second smaller random rectangle sometimes, following I-JEPA multi-block masking.
    if np.random.rand() < 0.5:
        small_h = max(1, target_h // 2)
        small_w = max(2, target_w // 2)
        sh = np.random.randint(0, max(1, grid_h - small_h + 1))
        sw = np.random.randint(0, max(1, grid_w - small_w + 1))
        mask[sh : sh + small_h, sw : sw + small_w] = True
        f0 = sh * stride
        f1 = min(mel.shape[0], f0 + patch_h + (small_h - 1) * stride)
        t0 = sw * stride
        t1 = min(mel.shape[1], t0 + patch_w + (small_w - 1) * stride)
        context[f0:f1, t0:t1] = 0.0

    return context.astype(np.float32), mask.reshape(-1)


class PatchEncoder(nn.Module):
    def __init__(self, cfg: AudioJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(
            1,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_stride,
        )
        grid_h = (cfg.n_mels - cfg.patch_size) // cfg.patch_stride + 1
        frames = 1 + int(cfg.sample_rate * cfg.duration_sec) // cfg.hop_length
        grid_w = (frames - cfg.patch_size) // cfg.patch_stride + 1
        self.num_patches = grid_h * grid_w
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.encoder_heads,
            dim_feedforward=cfg.embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.encoder_depth)
        self.norm = nn.LayerNorm(cfg.embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        pos = self.pos_embed[:, : tokens.shape[1], :]
        tokens = tokens + pos
        return self.norm(self.encoder(tokens))


class Predictor(nn.Module):
    def __init__(self, cfg: AudioJEPAConfig):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, cfg.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.predictor_heads,
            dim_feedforward=int(cfg.embed_dim * cfg.predictor_mlp_ratio),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=cfg.predictor_depth)
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, context_tokens: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        mask = target_mask.bool()
        x = context_tokens.clone()
        x[mask] = self.mask_token.expand(mask.sum(), -1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        return self.head(self.norm(self.net(x)))


class AudioJEPA(nn.Module):
    def __init__(self, cfg: AudioJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.online_encoder = PatchEncoder(cfg)
        self.target_encoder = PatchEncoder(cfg)
        self.predictor = Predictor(cfg)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    def forward(self, context: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> dict:
        context_tokens = self.online_encoder(context)
        with torch.no_grad():
            target_tokens = self.target_encoder(target)
        pred_tokens = self.predictor(context_tokens, target_mask)
        mask = target_mask.bool()
        pred = F.normalize(pred_tokens[mask], dim=-1)
        target_z = F.normalize(target_tokens[mask].detach(), dim=-1)
        pred_loss = 2.0 - 2.0 * (pred * target_z).sum(dim=-1).mean()
        var_loss = variance_loss(context_tokens, self.cfg.variance_floor)
        return {
            "loss": pred_loss + 0.05 * var_loss,
            "pred_loss": pred_loss.detach(),
            "var_loss": var_loss.detach(),
            "context_std": context_tokens.detach().std(dim=(0, 1)).mean(),
            "target_std": target_tokens.detach().std(dim=(0, 1)).mean(),
        }

    def update_target(self, momentum: float) -> None:
        with torch.no_grad():
            for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)


def variance_loss(tokens: torch.Tensor, floor: float) -> torch.Tensor:
    std = torch.sqrt(tokens.var(dim=(0, 1)) + 1e-4)
    return torch.mean(F.relu(floor - std))


def train_one_epoch(model: AudioJEPA, loader, optimizer, device, momentum: float, epoch: int) -> dict:
    model.train()
    losses = []
    pred_losses = []
    var_losses = []
    context_stds = []
    target_stds = []
    for context, target, target_mask in tqdm(loader, desc=f"audiojepa epoch {epoch}"):
        context = context.to(device)
        target = target.to(device)
        target_mask = target_mask.to(device)
        out = model(context, target, target_mask)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.update_target(momentum)
        losses.append(float(out["loss"].detach().cpu()))
        pred_losses.append(float(out["pred_loss"].cpu()))
        var_losses.append(float(out["var_loss"].cpu()))
        context_stds.append(float(out["context_std"].cpu()))
        target_stds.append(float(out["target_std"].cpu()))
    return {
        "epoch": epoch,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "pred_loss": float(np.mean(pred_losses)) if pred_losses else 0.0,
        "var_loss": float(np.mean(var_losses)) if var_losses else 0.0,
        "context_std": float(np.mean(context_stds)) if context_stds else 0.0,
        "target_std": float(np.mean(target_stds)) if target_stds else 0.0,
        "ema_momentum": momentum,
    }


def extract_embeddings(encoder: PatchEncoder, manifest: pd.DataFrame, project_root: Path, cfg: AudioJEPAConfig, device) -> np.ndarray:
    encoder.eval()
    vectors = []
    with torch.no_grad():
        for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="extract audiojepa"):
            path = resolve_audio_path(row.processed_path, project_root)
            y, _ = librosa.load(path, sr=cfg.sample_rate, mono=True)
            mel = make_logmel(y, cfg, random_crop=False, augment=False)
            x = torch.from_numpy(mel[None, None]).to(device)
            tokens = encoder(x)
            pooled = torch.cat([tokens.mean(dim=1), tokens.std(dim=1)], dim=1)
            vectors.append(pooled.cpu().numpy()[0])
    return np.vstack(vectors).astype(np.float32)


def cosine_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    t = step / (total_steps - 1)
    return end - (end - start) * (math.cos(math.pi * t) + 1.0) / 2.0


if __name__ == "__main__":
    main()

