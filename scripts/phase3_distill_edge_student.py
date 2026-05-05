"""Phase 3 edge-student knowledge distillation onto EfficientAT.

Distils the Phase 2A teacher ensemble into a real
EfficientAT MobileNetV3 student (mn04 / mn05 / mn10 / mn20) from
``github.com/fschmid56/EfficientAT``. The student keeps the AudioSet-pretrained
backbone — which already includes the ``baby_cry`` class — and gets a fresh
5-class head trained with:

  * Teacher soft labels from a per-seed validation-weighted Phase 2A ensemble.
  * KL divergence at temperature T plus label-smoothed cross-entropy.
  * Class-balanced batch sampler so every batch sees the rare classes.
  * SpecAugment via the EfficientAT ``AugmentMelSTFT`` mel front-end + mixup.
  * AdamW + cosine LR with warmup, early stop on validation macro-F1.
  * Per-seed resumable checkpoints so a Colab disconnect does not waste work.

Inputs are resampled 16 kHz → 32 kHz on the fly because EfficientAT was
pretrained at 32 kHz. The script defaults to 5-seed repeated-split evaluation
to match Phase 2A's reporting protocol; it writes:

  * results/phase2a/metrics/phase3_edge_student_<variant>_repeated_eval.json
  * results/phase2a/metrics/phase3_edge_student_<variant>_metrics.json
    (single-best-seed detail, INT8 footprint, CPU/GPU latency)
  * results/phase2a/predictions/phase3_edge_student_<variant>_test_predictions.csv
    (predictions from the best-seed run)
  * results/phase2a/artifacts/phase3_edge_student_<variant>_{fp32,int8}.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2a.config import (
    CLASSES,
    FEATURES_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    SAMPLE_RATE,
    SUPERVISED_MANIFEST,
    TEST_SIZE,
    VAL_SIZE,
)
from src.phase2a.data import load_audio, load_manifest, resolve_audio_path


EFFICIENTAT_REPO = "https://github.com/fschmid56/EfficientAT.git"
EFFICIENTAT_CACHE = PROJECT_ROOT / ".cache" / "EfficientAT"
DEFAULT_TEACHER_BANKS = [
    FEATURES_DIR / "ablation_aux_with_whisper_features.npz",
    FEATURES_DIR / "ablation_aux_no_whisper_features.npz",
    FEATURES_DIR / "ablation_no_aux_with_whisper_features.npz",
    FEATURES_DIR / "ast_embeddings.npz",
    FEATURES_DIR / "ast_aux_adapted_embeddings.npz",
    FEATURES_DIR / "whisper_embeddings.npz",
    FEATURES_DIR / "handcrafted_features.npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-bank",
        type=Path,
        default=FEATURES_DIR / "ablation_aux_with_whisper_features.npz",
        help="Primary cached feature bank used for manifest/sample ordering.",
    )
    parser.add_argument(
        "--teacher-banks",
        type=Path,
        nargs="*",
        default=DEFAULT_TEACHER_BANKS,
        help="Feature banks used to build the per-seed validation-weighted teacher ensemble.",
    )
    parser.add_argument("--manifest", type=Path, default=SUPERVISED_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "metrics")
    parser.add_argument("--checkpoint-dir", type=Path, default=RESULTS_DIR / "artifacts")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument(
        "--variant",
        type=str,
        default="mn10_as",
        choices=["mn04_as", "mn05_as", "mn10_as", "mn20_as"],
        help="EfficientAT pretrained variant. mn04=0.98M, mn05=1.43M, mn10=4.88M, mn20=17.91M.",
    )
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of repeated stratified splits.")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr-mult", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kd-alpha", type=float, default=0.7)
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--svm-c", type=float, default=3.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--svm-pca", type=int, default=128)
    parser.add_argument("--teacher-top-n", type=int, default=8)
    parser.add_argument("--teacher-random-candidates", type=int, default=1500)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-sample-rate", type=int, default=32000)
    parser.add_argument("--efficientat-cache", type=Path, default=EFFICIENTAT_CACHE)
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable per-seed checkpoint resume.")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_efficientat(cache_dir: Path) -> None:
    """Clone EfficientAT into a local cache and add it to sys.path."""
    if not (cache_dir / "models").exists():
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning EfficientAT into {cache_dir}")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", EFFICIENTAT_REPO, str(cache_dir)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    if str(cache_dir) not in sys.path:
        sys.path.insert(0, str(cache_dir))
    metadata_path = cache_dir / "metadata" / "class_labels_indices.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"EfficientAT clone is incomplete: missing {metadata_path}. "
            "Delete the .cache/EfficientAT folder and rerun this script so it can clone a clean copy."
        )


@contextlib.contextmanager
def efficientat_workdir(cache_dir: Path):
    """Run EfficientAT imports/builds from its repo root.

    EfficientAT's helpers open files like ``metadata/class_labels_indices.csv``
    relative to the current working directory, so imports fail unless cwd is the
    cloned repository root.
    """
    old_cwd = Path.cwd()
    os.chdir(cache_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def load_feature_bank(path: Path):
    data = np.load(path, allow_pickle=True)
    return (
        data["X"].astype(np.float32),
        data["y"].astype(np.int64),
        data["sample_ids"].astype(str),
    )


def load_teacher_banks(paths: list[Path], reference_sample_ids: np.ndarray, reference_y: np.ndarray) -> dict[str, dict]:
    """Load and align all available Phase 2A feature banks to a reference order."""
    banks: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            print(f"Teacher bank missing, skipping: {path.name}")
            continue
        X, y, sample_ids = load_feature_bank(path)
        index = {sid: idx for idx, sid in enumerate(sample_ids)}
        if any(sid not in index for sid in reference_sample_ids):
            print(f"Teacher bank incomplete, skipping: {path.name}")
            continue
        order = np.asarray([index[sid] for sid in reference_sample_ids])
        aligned_y = y[order]
        if not np.array_equal(aligned_y, reference_y):
            raise RuntimeError(f"Label mismatch in teacher bank: {path}")
        feature_name = path.stem.replace("_features", "").replace("_embeddings", "")
        banks[feature_name] = {"path": path, "X": X[order].astype(np.float32)}
    if not banks:
        raise RuntimeError("No usable teacher feature banks found.")
    return banks


def stratified_three_split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, stratify=labels, random_state=seed)
    relative_val = VAL_SIZE / (1.0 - TEST_SIZE)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=relative_val, stratify=labels[trainval_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


def build_rbf_svm_pipeline(
    n_features: int,
    c_value: float = 3.0,
    gamma: str | float = "scale",
    pca_components: int | None = 128,
    use_l2: bool = False,
    probability: bool = True,
):
    """Local copy of the Phase 2B helper so Phase 3 works from a Phase 2A-only package."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    from sklearn.svm import SVC

    steps = [("scaler", StandardScaler())]
    if pca_components is not None and pca_components < n_features:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
    if use_l2:
        steps.append(("l2", Normalizer()))
    steps.append(
        (
            "clf",
            SVC(
                kernel="rbf",
                C=c_value,
                gamma=gamma,
                class_weight="balanced",
                probability=probability,
                random_state=42,
            ),
        )
    )
    return Pipeline(steps)


def fit_teacher_ensemble_for_seed(
    teacher_banks: dict[str, dict],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Build a strong validation-weighted teacher from all cached Phase 2A views."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    candidates = []
    for bank_name, bank in teacher_banks.items():
        X = bank["X"]
        models = [
            (
                f"{bank_name}::rbf_svm_balanced",
                build_rbf_svm_pipeline(
                    X.shape[1],
                    c_value=args.svm_c,
                    gamma=args.svm_gamma,
                    pca_components=min(args.svm_pca, X.shape[1] - 1) if args.svm_pca else None,
                    use_l2=False,
                    probability=True,
                ),
            ),
            (
                f"{bank_name}::logreg_balanced",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                C=1.0,
                                class_weight="balanced",
                                max_iter=3000,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            ),
        ]
        for model_name, model in models:
            try:
                model.fit(X[train_idx], y[train_idx])
                probs = model.predict_proba(X).astype(np.float32)
                val_macro_f1 = f1_score(y[val_idx], probs[val_idx].argmax(axis=1), average="macro", zero_division=0)
                candidates.append({"name": model_name, "probs": probs, "val_macro_f1": float(val_macro_f1)})
            except Exception as exc:
                print(f"Teacher candidate failed ({model_name}): {exc}")

    if not candidates:
        raise RuntimeError("No teacher candidates were successfully trained.")

    candidates = sorted(candidates, key=lambda row: row["val_macro_f1"], reverse=True)[: args.teacher_top_n]
    weights = search_teacher_weights(
        [row["probs"][val_idx] for row in candidates],
        y[val_idx],
        random_candidates=args.teacher_random_candidates,
        seed=seed,
    )
    probs = weighted_average([row["probs"] for row in candidates], weights).astype(np.float32)
    meta = {
        "selected_teachers": [row["name"] for row in candidates],
        "selected_val_macro_f1": [row["val_macro_f1"] for row in candidates],
        "weights": weights.tolist(),
        "ensemble_val_macro_f1": float(f1_score(y[val_idx], probs[val_idx].argmax(axis=1), average="macro", zero_division=0)),
    }
    return probs, meta


def search_teacher_weights(probs_list: list[np.ndarray], y: np.ndarray, random_candidates: int, seed: int) -> np.ndarray:
    from sklearn.metrics import f1_score

    n = len(probs_list)
    if n == 1:
        return np.ones(1, dtype=np.float32)
    rng = np.random.default_rng(seed)
    candidates = [np.ones(n) / n]
    for idx in range(n):
        one_hot = np.zeros(n)
        one_hot[idx] = 1.0
        candidates.append(one_hot)
    val_scores = np.asarray([f1_score(y, p.argmax(axis=1), average="macro", zero_division=0) for p in probs_list])
    score_weights = np.maximum(val_scores, 1e-6)
    candidates.append(score_weights / score_weights.sum())
    for alpha in [0.25, 0.5, 1.0, 2.0]:
        candidates.extend(rng.dirichlet(np.ones(n) * alpha, size=max(1, random_candidates // 4)))

    best = candidates[0]
    best_score = -1.0
    for weights in candidates:
        score = f1_score(y, weighted_average(probs_list, weights).argmax(axis=1), average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best = weights
    return np.asarray(best, dtype=np.float32)


def weighted_average(probs_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, np.stack(probs_list, axis=0), axes=(0, 0))


def build_student(variant: str):
    """Real EfficientAT MobileNetV3 student with replaced 5-class head."""
    import io

    import torch.nn as nn

    with efficientat_workdir(EFFICIENTAT_CACHE):
        from helpers.utils import NAME_TO_WIDTH
        from models.mn.model import get_model as get_mn

        width = NAME_TO_WIDTH(variant)
        # EfficientAT prints the whole model at construction; keep Colab logs readable.
        with contextlib.redirect_stdout(io.StringIO()):
            model = get_mn(width_mult=width, pretrained_name=variant, head_type="mlp", num_classes=len(CLASSES))
    head = getattr(model, "classifier", None)
    replaced = False
    if isinstance(head, nn.Sequential):
        for idx in range(len(head) - 1, -1, -1):
            layer = head[idx]
            if isinstance(layer, nn.Linear):
                head[idx] = nn.Linear(layer.in_features, len(CLASSES))
                replaced = True
                break
    elif isinstance(head, nn.Linear):
        model.classifier = nn.Linear(head.in_features, len(CLASSES))
        replaced = True
    if not replaced:
        raise RuntimeError("Could not locate EfficientAT classifier head to replace for 5-class output.")
    return model


def build_mel_frontend(target_sr: int):
    """EfficientAT's AugmentMelSTFT log-mel front-end (32 kHz default)."""
    with efficientat_workdir(EFFICIENTAT_CACHE):
        from models.preprocess import AugmentMelSTFT

    return AugmentMelSTFT(n_mels=128, sr=target_sr, win_length=800, hopsize=320)


class Resampler:
    def __init__(self, src_sr: int, dst_sr: int, device):
        import torchaudio

        self.transform = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr).to(device)

    def __call__(self, audio):
        return self.transform(audio)


def spec_augment_mask(mel, freq_mask: int = 16, time_mask: int = 32, num_freq: int = 1, num_time: int = 2):
    import torch

    mel = mel.clone()
    if mel.dim() == 3:
        mel = mel.unsqueeze(1)
    _B, _C, F, T = mel.shape
    for _ in range(num_freq):
        f = int(torch.randint(0, freq_mask + 1, (1,)).item())
        if f == 0:
            continue
        f0 = int(torch.randint(0, max(1, F - f), (1,)).item())
        mel[:, :, f0 : f0 + f, :] = 0.0
    for _ in range(num_time):
        t = int(torch.randint(0, time_mask + 1, (1,)).item())
        if t == 0:
            continue
        t0 = int(torch.randint(0, max(1, T - t), (1,)).item())
        mel[:, :, :, t0 : t0 + t] = 0.0
    return mel


def mixup(audio, labels_oh, teacher_probs, alpha: float):
    import torch

    if alpha <= 0:
        return audio, labels_oh, teacher_probs
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(audio.size(0), device=audio.device)
    audio_mix = lam * audio + (1 - lam) * audio[perm]
    labels_mix = lam * labels_oh + (1 - lam) * labels_oh[perm]
    teacher_mix = lam * teacher_probs + (1 - lam) * teacher_probs[perm]
    return audio_mix, labels_mix, teacher_mix


def distill_loss(student_logits, teacher_probs_soft, labels_oh, T: float, alpha: float, label_smoothing: float):
    import torch
    import torch.nn.functional as F

    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    kl = F.kl_div(log_p_student, teacher_probs_soft, reduction="batchmean") * (T * T)
    n_classes = labels_oh.size(-1)
    smooth = labels_oh * (1.0 - label_smoothing) + label_smoothing / n_classes
    log_p_full = F.log_softmax(student_logits, dim=-1)
    ce = -(smooth * log_p_full).sum(dim=-1).mean()
    return alpha * kl + (1.0 - alpha) * ce, float(kl.detach().item()), float(ce.detach().item())


class CryAudioDataset:
    def __init__(self, manifest_rows: pd.DataFrame, sample_ids_to_probs: dict[str, np.ndarray], project_root: Path):
        self.rows = manifest_rows.reset_index(drop=True)
        self.id_to_probs = sample_ids_to_probs
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        import torch

        row = self.rows.iloc[idx]
        audio = load_audio(resolve_audio_path(row["processed_path"], self.project_root))
        teacher = self.id_to_probs[str(row["sample_id"])]
        return (
            torch.from_numpy(audio).float(),
            int(row["label_idx"]),
            torch.from_numpy(teacher).float(),
        )


def collate(batch):
    import torch

    audios, labels, teacher = zip(*batch)
    return torch.stack(audios), torch.tensor(labels, dtype=torch.long), torch.stack(teacher)


def class_balanced_sampler(labels: np.ndarray):
    from torch.utils.data import WeightedRandomSampler

    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(weights.tolist(), num_samples=len(labels), replacement=True)


def forward_logits(model, mel_input):
    out = model(mel_input)
    if isinstance(out, tuple):
        return out[0]
    return out


def evaluate(model, loader, mel, resampler, device):
    import torch

    model.eval()
    preds, probs, truths = [], [], []
    with torch.no_grad():
        for audio, labels, _teacher in loader:
            audio = audio.to(device)
            spec = mel(resampler(audio))
            if spec.dim() == 3:
                spec = spec.unsqueeze(1)
            logits = forward_logits(model, spec)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            preds.append(p.argmax(axis=1))
            probs.append(p)
            truths.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(truths)


def metrics_from(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict:
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    report = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0)
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASSES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist(),
    }
    if probs is not None:
        confidence = probs.max(axis=1)
        out["confidence_mean"] = float(confidence.mean())
        out["ece"] = expected_calibration_error(y_true, y_pred, confidence)
        out["abstention"] = abstention_curve(y_true, y_pred, confidence)
    return out


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    correct = (y_true == y_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        upper = confidence <= hi if hi == 1.0 else confidence < hi
        mask = (confidence >= lo) & upper
        if mask.sum() == 0:
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def abstention_curve(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> dict[str, dict]:
    out = {}
    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = confidence >= threshold
        if mask.sum() == 0:
            out[str(threshold)] = {"coverage": 0.0, "accuracy": None, "macro_f1": None}
            continue
        from sklearn.metrics import f1_score

        out[str(threshold)] = {
            "coverage": float(mask.mean()),
            "accuracy": float((y_true[mask] == y_pred[mask]).mean()),
            "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
        }
    return out


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def split_param_groups(model, head_lr_mult: float, base_lr: float):
    head_params, backbone_params = [], []
    head = getattr(model, "classifier", None)
    head_param_ids = set()
    if head is not None:
        for p in head.parameters():
            head_param_ids.add(id(p))
            head_params.append(p)
    for p in model.parameters():
        if id(p) not in head_param_ids and p.requires_grad:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": base_lr * head_lr_mult},
    ]


def measure_latency(model, mel, resampler, device, n_warmup: int = 5, n_runs: int = 30) -> float:
    import torch

    model.eval()
    audio = torch.zeros(1, int(SAMPLE_RATE * 10.0), device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            spec = mel(resampler(audio))
            if spec.dim() == 3:
                spec = spec.unsqueeze(1)
            _ = forward_logits(model, spec)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            spec = mel(resampler(audio))
            if spec.dim() == 3:
                spec = spec.unsqueeze(1)
            _ = forward_logits(model, spec)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / n_runs * 1000.0


def quantise_int8(model_cpu):
    import torch
    import torch.nn as nn

    try:
        return torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
    except Exception as exc:
        print(f"Dynamic quantisation failed ({exc}); returning fp32 model.")
        return model_cpu


def run_one_seed(
    seed: int,
    df: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    teacher_banks: dict[str, dict],
    args: argparse.Namespace,
    device,
):
    """Train, validate, test the student for one seed; return per-seed records."""
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    set_seed(seed)
    train_idx, val_idx, test_idx = stratified_three_split(y, seed)

    teacher_probs, teacher_meta = fit_teacher_ensemble_for_seed(teacher_banks, y, train_idx, val_idx, args, seed)
    teacher_test_pred = teacher_probs[test_idx].argmax(axis=1)
    teacher_test_metrics = metrics_from(y[test_idx], teacher_test_pred, teacher_probs[test_idx])

    id_to_probs = {sid: teacher_probs[i] for i, sid in enumerate(sample_ids)}
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    train_ids = sample_ids[train_idx]
    val_ids = sample_ids[val_idx]
    test_ids = sample_ids[test_idx]
    df_indexed = df.set_index("sample_id", drop=False)
    train_rows = df_indexed.loc[train_ids].reset_index(drop=True)
    val_rows = df_indexed.loc[val_ids].reset_index(drop=True)
    test_rows = df_indexed.loc[test_ids].reset_index(drop=True)

    train_ds = CryAudioDataset(train_rows, id_to_probs, PROJECT_ROOT)
    val_ds = CryAudioDataset(val_rows, id_to_probs, PROJECT_ROOT)
    test_ds = CryAudioDataset(test_rows, id_to_probs, PROJECT_ROOT)
    sampler = class_balanced_sampler(train_rows["label_idx"].to_numpy())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    model = build_student(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    mel = build_mel_frontend(args.target_sample_rate).to(device)
    mel.eval()
    resampler = Resampler(SAMPLE_RATE, args.target_sample_rate, device)
    optim = AdamW(split_param_groups(model, args.head_lr_mult, args.lr), weight_decay=args.weight_decay)

    history = []
    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    start_epoch = 0
    stale_epochs = 0
    seed_ckpt_path = args.checkpoint_dir / f"phase3_edge_student_{args.variant}_seed_{seed}.partial.pt"
    if args.resume and seed_ckpt_path.exists():
        ckpt = torch.load(seed_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optimizer_state"])
        best_state = {k: v.cpu().clone() for k, v in ckpt["best_state"].items()} if ckpt.get("best_state") else None
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        history = ckpt.get("history", [])
        start_epoch = int(ckpt.get("epoch", 0))
        stale_epochs = int(ckpt.get("stale_epochs", 0))
        print(f"  resumed seed checkpoint from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(epoch, args.epochs, args.warmup_epochs, args.lr)
        optim.param_groups[0]["lr"] = lr
        optim.param_groups[1]["lr"] = lr * args.head_lr_mult
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for audio, lbl, teacher in train_loader:
            audio = audio.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            labels_oh = F.one_hot(lbl, num_classes=len(CLASSES)).float()
            audio_aug, labels_mix, teacher_mix = mixup(audio, labels_oh, teacher, args.mixup_alpha)
            spec = mel(resampler(audio_aug))
            if spec.dim() == 3:
                spec = spec.unsqueeze(1)
            spec = spec_augment_mask(spec)
            logits = forward_logits(model, spec)
            teacher_soft = F.softmax(torch.log(teacher_mix.clamp(min=1e-8)) / args.kd_temperature, dim=-1)
            loss, _kl, _ce = distill_loss(logits, teacher_soft, labels_mix, args.kd_temperature, args.kd_alpha, args.label_smoothing)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            ep_loss += loss.item()
            n_batches += 1
        val_pred, _val_probs, val_truth = evaluate(model, val_loader, mel, resampler, device)
        val_metrics = metrics_from(val_truth, val_pred)
        history.append({"epoch": epoch + 1, "train_loss": ep_loss / max(1, n_batches), "val_macro_f1": val_metrics["macro_f1"]})
        if val_metrics["macro_f1"] > best_val_f1 + 1e-4:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "best_state": best_state,
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "history": history,
                "stale_epochs": stale_epochs,
                "teacher_meta": teacher_meta,
            },
            seed_ckpt_path,
        )
        if stale_epochs >= args.early_stop_patience:
            print(f"  early stop at epoch {epoch + 1}; best epoch={best_epoch}, best val macro-F1={best_val_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, val_probs, val_truth = evaluate(model, val_loader, mel, resampler, device)
    val_metrics = metrics_from(val_truth, val_pred, val_probs)
    test_pred, test_probs, test_truth = evaluate(model, test_loader, mel, resampler, device)
    test_metrics = metrics_from(test_truth, test_pred, test_probs)
    agreement = float((teacher_test_pred == test_pred).mean())

    return {
        "seed": int(seed),
        "n_params": int(n_params),
        "best_val_macro_f1": float(best_val_f1),
        "validation": val_metrics,
        "test": test_metrics,
        "teacher_test": teacher_test_metrics,
        "teacher_meta": teacher_meta,
        "teacher_student_agreement": agreement,
        "best_epoch": best_epoch,
        "history": history,
        "test_sample_ids": test_ids.tolist(),
        "test_truth": test_truth.tolist(),
        "test_pred": test_pred.tolist(),
        "test_probs": test_probs.tolist(),
        "best_state": best_state,
    }


def aggregate(per_seed: list[dict]) -> dict:
    def collect(path: list[str]) -> list[float]:
        out = []
        for r in per_seed:
            v = r
            for key in path:
                v = v[key]
            out.append(float(v))
        return out

    def stats(values: list[float]) -> dict:
        arr = np.asarray(values)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

    summary = {
        "test_macro_f1": stats(collect(["test", "macro_f1"])),
        "test_balanced_accuracy": stats(collect(["test", "balanced_accuracy"])),
        "test_weighted_f1": stats(collect(["test", "weighted_f1"])),
        "test_accuracy": stats(collect(["test", "accuracy"])),
        "test_ece": stats(collect(["test", "ece"])),
        "val_macro_f1": stats(collect(["validation", "macro_f1"])),
        "teacher_test_macro_f1": stats(collect(["teacher_test", "macro_f1"])),
        "teacher_test_ece": stats(collect(["teacher_test", "ece"])),
        "teacher_student_agreement": stats(collect(["teacher_student_agreement"])),
    }
    summary["macro_f1_retention_pct"] = stats(
        [
            r["test"]["macro_f1"] / max(r["teacher_test"]["macro_f1"], 1e-8) * 100.0
            for r in per_seed
        ]
    )
    per_class_acc: dict[str, list[float]] = {}
    for cls in CLASSES:
        per_class_acc[cls] = [r["test"]["per_class"][cls]["f1"] for r in per_seed]
    summary["per_class_test_f1"] = {cls: stats(values) for cls, values in per_class_acc.items()}
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    ensure_efficientat(args.efficientat_cache)

    import torch

    df = load_manifest(args.manifest)
    df["sample_id"] = df["sample_id"].astype(str)

    X, y, sample_ids = load_feature_bank(args.feature_bank)
    df_indexed = df.set_index("sample_id", drop=False)
    aligned = df_indexed.loc[sample_ids].reset_index(drop=True)
    if not (aligned["label_idx"].to_numpy() == y).all():
        raise RuntimeError("Manifest labels do not match cached feature bank labels.")
    teacher_banks = load_teacher_banks(args.teacher_banks, sample_ids, y)

    rng = np.random.default_rng(args.seed_base)
    seeds = [int(s) for s in rng.integers(0, 100_000, size=args.n_seeds)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Variant:        {args.variant}")
    print(f"Device:         {device}")
    print(f"Feature bank:   {args.feature_bank.name}")
    print(f"Teacher banks:  {', '.join(teacher_banks)}")
    print(f"Seeds:          {seeds}")
    print(f"Epochs / seed:  {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print()

    per_seed_records: list[dict] = []
    best_seed = None
    for seed in seeds:
        print(f"=== seed={seed} ===")
        record = run_one_seed(seed, aligned, y, sample_ids, teacher_banks, args, device)
        per_seed_records.append(record)
        print(
            f"  test_macro_f1={record['test']['macro_f1']:.4f} "
            f"val_macro_f1={record['validation']['macro_f1']:.4f} "
            f"teacher_test_macro_f1={record['teacher_test']['macro_f1']:.4f} "
            f"agreement={record['teacher_student_agreement']*100:.1f}%"
        )
        if best_seed is None or record["best_val_macro_f1"] > best_seed["best_val_macro_f1"]:
            best_seed = record

    summary = aggregate(per_seed_records)

    runs_for_json = []
    for r in per_seed_records:
        rr = {k: v for k, v in r.items() if k != "best_state"}
        rr["test_probs"] = [list(map(float, row)) for row in r["test_probs"]]
        runs_for_json.append(rr)

    repeated_eval_path = args.output_dir / f"phase3_edge_student_{args.variant}_repeated_eval.json"
    repeated_eval_path.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "feature_bank": str(args.feature_bank),
                "teacher_banks": {name: str(bank["path"]) for name, bank in teacher_banks.items()},
                "n_seeds": args.n_seeds,
                "seeds": seeds,
                "summary": summary,
                "runs": runs_for_json,
                "hyperparameters": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "head_lr_mult": args.head_lr_mult,
                    "weight_decay": args.weight_decay,
                    "kd_alpha": args.kd_alpha,
                    "kd_temperature": args.kd_temperature,
                    "label_smoothing": args.label_smoothing,
                    "mixup_alpha": args.mixup_alpha,
                    "target_sample_rate": args.target_sample_rate,
                    "teacher_top_n": args.teacher_top_n,
                    "teacher_random_candidates": args.teacher_random_candidates,
                    "early_stop_patience": args.early_stop_patience,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cpu_device = torch.device("cpu")
    final_model = build_student(args.variant)
    final_model.load_state_dict(best_seed["best_state"])
    final_model.eval()
    fp32_path = args.checkpoint_dir / f"phase3_edge_student_{args.variant}_fp32.pt"
    torch.save({"state_dict": best_seed["best_state"], "variant": args.variant, "best_seed": best_seed["seed"]}, fp32_path)
    fp32_size_kb = fp32_path.stat().st_size / 1024.0
    quantised = quantise_int8(final_model)
    int8_path = args.checkpoint_dir / f"phase3_edge_student_{args.variant}_int8.pt"
    torch.save(quantised.state_dict(), int8_path)
    int8_size_kb = int8_path.stat().st_size / 1024.0

    cpu_mel = build_mel_frontend(args.target_sample_rate).to(cpu_device)
    cpu_mel.eval()
    cpu_resampler = Resampler(SAMPLE_RATE, args.target_sample_rate, cpu_device)
    fp32_latency_ms = measure_latency(final_model.to(cpu_device), cpu_mel, cpu_resampler, cpu_device)
    try:
        int8_latency_ms = measure_latency(quantised.to(cpu_device), cpu_mel, cpu_resampler, cpu_device)
    except Exception:
        int8_latency_ms = float("nan")
    if device.type == "cuda":
        gpu_mel = build_mel_frontend(args.target_sample_rate).to(device)
        gpu_mel.eval()
        gpu_resampler = Resampler(SAMPLE_RATE, args.target_sample_rate, device)
        gpu_model = build_student(args.variant).to(device)
        gpu_model.load_state_dict(best_seed["best_state"])
        gpu_latency_ms = measure_latency(gpu_model, gpu_mel, gpu_resampler, device)
    else:
        gpu_latency_ms = float("nan")

    pred_df = pd.DataFrame(
        {
            "sample_id": best_seed["test_sample_ids"],
            "true_idx": best_seed["test_truth"],
            "pred_idx": best_seed["test_pred"],
        }
    )
    test_probs_arr = np.asarray(best_seed["test_probs"])
    pred_df["confidence"] = test_probs_arr.max(axis=1)
    for idx, name in enumerate(CLASSES):
        pred_df[f"prob_{name}"] = test_probs_arr[:, idx]
    pred_path = args.predictions_dir / f"phase3_edge_student_{args.variant}_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = args.output_dir / f"phase3_edge_student_{args.variant}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "best_seed": best_seed["seed"],
                "n_params": best_seed["n_params"],
                "fp32_size_kb": float(fp32_size_kb),
                "int8_size_kb": float(int8_size_kb),
                "fp32_cpu_latency_ms": float(fp32_latency_ms),
                "int8_cpu_latency_ms": float(int8_latency_ms),
                "gpu_latency_ms": float(gpu_latency_ms),
                "best_val_macro_f1": best_seed["best_val_macro_f1"],
                "best_test_metrics": best_seed["test"],
                "best_validation_metrics": best_seed["validation"],
                "best_seed_teacher_metrics": best_seed["teacher_test"],
                "best_seed_teacher_student_agreement": best_seed["teacher_student_agreement"],
                "training_history": best_seed["history"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    legacy_path = args.output_dir / "phase3_edge_student_metrics.json"
    legacy_path.write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")

    print()
    print("=== Phase 3 Edge Distillation — Aggregate Summary ===")
    print(f"Variant:                   {args.variant}")
    print(f"Seeds:                     {args.n_seeds}")
    print(
        "Test macro-F1:             "
        f"{summary['test_macro_f1']['mean']:.4f} ± {summary['test_macro_f1']['std']:.4f} "
        f"(min {summary['test_macro_f1']['min']:.4f}, max {summary['test_macro_f1']['max']:.4f})"
    )
    print(
        "Teacher test macro-F1:     "
        f"{summary['teacher_test_macro_f1']['mean']:.4f} ± {summary['teacher_test_macro_f1']['std']:.4f}"
    )
    print(
        "Macro-F1 retention vs T:   "
        f"{summary['macro_f1_retention_pct']['mean']:.1f}% ± {summary['macro_f1_retention_pct']['std']:.1f}%"
    )
    print(
        "Teacher/student agreement: "
        f"{summary['teacher_student_agreement']['mean']*100:.1f}% ± {summary['teacher_student_agreement']['std']*100:.1f}%"
    )
    print(f"Student ECE:               {summary['test_ece']['mean']:.4f} ± {summary['test_ece']['std']:.4f}")
    print(f"Teacher ECE:               {summary['teacher_test_ece']['mean']:.4f} ± {summary['teacher_test_ece']['std']:.4f}")
    for cls in CLASSES:
        s = summary["per_class_test_f1"][cls]
        print(f"  per-class F1[{cls:<11}]: {s['mean']:.3f} ± {s['std']:.3f}")
    print()
    print(f"Best-seed footprint: params={best_seed['n_params']:,} fp32={fp32_size_kb:.1f}KB int8={int8_size_kb:.1f}KB")
    print(f"Best-seed latency:   cpu_fp32={fp32_latency_ms:.1f}ms cpu_int8={int8_latency_ms:.1f}ms gpu={gpu_latency_ms:.1f}ms")
    print(f"Saved repeated-eval: {repeated_eval_path}")
    print(f"Saved metrics:       {metrics_path}")
    print(f"Saved predictions:   {pred_path}")
    print(f"Saved checkpoints:   {fp32_path}, {int8_path}")


if __name__ == "__main__":
    main()
