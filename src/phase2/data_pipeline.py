"""Data loading, infant-ID splitting, augmentation, and PyTorch datasets."""

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

from src.phase2.config import (
    RAW_DIR, SAMPLE_RATE, DURATION, N_SAMPLES,
    CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE,
    AUG_TARGET_PER_CLASS, AUG_NOISE_AMP, AUG_TIME_SHIFT_SEC,
    AUG_STRETCH_RANGE, AUG_SNR_RANGE,
    SPEC_AUG_FREQ_MASK, SPEC_AUG_TIME_MASK, SPEC_AUG_NUM_MASKS,
    MIXUP_ALPHA, MEL_N_MELS,
)
from src.phase2.features import extract_mel_spectrogram, extract_domain_features


# ── Infant ID parsing ────────────────────────────────────────────────

def parse_infant_id(filepath):
    """Extract infant UUID from Donate-a-Cry filename.
    Format: {UUID(36 chars)}-{timestamp}-{version}-{gender}-{age}-{reason}.wav
    """
    stem = Path(filepath).stem
    if "_aug_" in stem:
        stem = stem.split("_aug_")[0]
    return stem[:36].lower()


# ── Audio loading ────────────────────────────────────────────────────

def load_audio(filepath, sr=SAMPLE_RATE, duration=DURATION):
    """Load and fix-length a single audio file."""
    y, _ = librosa.load(filepath, sr=sr, duration=duration, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def load_all_originals(data_dir=None):
    """Load all original (non-augmented) audio files.
    Returns list of dicts with audio, label, label_idx, filepath, infant_id.
    """
    if data_dir is None:
        data_dir = RAW_DIR
    data_dir = Path(data_dir)
    dataset = []
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            print(f"[WARN] Missing directory: {cls_dir}")
            continue
        files = sorted(
            f for f in cls_dir.iterdir()
            if f.suffix.lower() == ".wav" and "_aug_" not in f.stem
        )
        print(f"  {cls}: {len(files)} originals")
        for fp in files:
            try:
                waveform = load_audio(fp)
                dataset.append({
                    "audio": waveform,
                    "label": cls,
                    "label_idx": CLASS_TO_IDX[cls],
                    "filepath": str(fp),
                    "infant_id": parse_infant_id(fp),
                })
            except Exception as e:
                print(f"  [ERROR] {fp.name}: {e}")
    print(f"Total: {len(dataset)} originals loaded")
    return dataset


# ── Group-wise split by infant ID ────────────────────────────────────

def group_split(dataset):
    """Split dataset by infant ID into train/val/test (70/15/15).
    Ensures no infant appears in multiple splits.
    """
    infant_ids = np.array([d["infant_id"] for d in dataset])
    labels = np.array([d["label_idx"] for d in dataset])
    indices = np.arange(len(dataset))

    unique_ids = np.unique(infant_ids)
    print(f"Unique infant IDs: {len(unique_ids)}")

    # First split: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss1.split(indices, labels, groups=infant_ids))

    # Second split: train vs val (from the trainval portion)
    relative_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    trainval_ids = infant_ids[trainval_idx]
    trainval_labels = labels[trainval_idx]
    trainval_indices = np.arange(len(trainval_idx))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=RANDOM_STATE)
    train_sub, val_sub = next(gss2.split(trainval_indices, trainval_labels, groups=trainval_ids))

    train_idx_final = trainval_idx[train_sub]
    val_idx_final = trainval_idx[val_sub]

    train = [dataset[i] for i in train_idx_final]
    val = [dataset[i] for i in val_idx_final]
    test = [dataset[i] for i in test_idx]

    # Verify no infant leakage
    train_ids = set(d["infant_id"] for d in train)
    val_ids = set(d["infant_id"] for d in val)
    test_ids = set(d["infant_id"] for d in test)
    assert len(train_ids & val_ids) == 0, "Infant leak: train/val overlap"
    assert len(train_ids & test_ids) == 0, "Infant leak: train/test overlap"
    assert len(val_ids & test_ids) == 0, "Infant leak: val/test overlap"

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        dist = Counter(d["label"] for d in split)
        print(f"  {name}: {len(split)} samples — {dict(sorted(dist.items()))}")

    return train, val, test


# ── Waveform augmentation ────────────────────────────────────────────

def _add_gaussian_noise(y):
    amp = np.random.uniform(*AUG_NOISE_AMP)
    return (y + amp * np.random.randn(len(y))).astype(np.float32)


def _time_shift(y, sr=SAMPLE_RATE):
    shift = int(sr * np.random.uniform(-AUG_TIME_SHIFT_SEC, AUG_TIME_SHIFT_SEC))
    shifted = np.roll(y, shift)
    if shift > 0:
        shifted[:shift] = 0
    elif shift < 0:
        shifted[shift:] = 0
    return shifted


def _time_stretch(y):
    rate = np.random.uniform(*AUG_STRETCH_RANGE)
    stretched = librosa.effects.time_stretch(y=y, rate=rate)
    target = len(y)
    if len(stretched) < target:
        stretched = np.pad(stretched, (0, target - len(stretched)))
    else:
        stretched = stretched[:target]
    return stretched.astype(np.float32)


def _pitch_shift(y, sr=SAMPLE_RATE):
    steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps).astype(np.float32)


def _reverb_sim(y, sr=SAMPLE_RATE):
    rt60 = np.random.uniform(0.1, 0.3)
    n = int(sr * rt60)
    impulse = np.random.randn(n) * np.exp(-np.linspace(0, 5, n))
    impulse /= np.max(np.abs(impulse)) + 1e-8
    wet = np.random.uniform(0.1, 0.3)
    y_rev = np.convolve(y, impulse, mode="full")[:len(y)]
    return ((1 - wet) * y + wet * y_rev).astype(np.float32)


def _add_background_noise(y, sr=SAMPLE_RATE):
    """Mix with pink noise at random SNR."""
    from scipy.signal import lfilter
    snr_db = np.random.uniform(*AUG_SNR_RANGE)
    noise = np.random.randn(len(y)).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    noise = lfilter(b, a, noise).astype(np.float32)
    sig_power = np.mean(y ** 2)
    noise_power = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    return (y + scale * noise).astype(np.float32)


WAVEFORM_AUGS = [
    _add_gaussian_noise, _time_shift, _time_stretch,
    _pitch_shift, _reverb_sim, _add_background_noise,
]


def augment_waveform(y):
    """Apply a random composition of 2-3 augmentations."""
    n_augs = np.random.randint(2, 4)
    chosen = np.random.choice(len(WAVEFORM_AUGS), size=n_augs, replace=False)
    for idx in chosen:
        y = WAVEFORM_AUGS[idx](y)
    return y


def augment_training_set(train_data, target=AUG_TARGET_PER_CLASS):
    """Augment minority classes in training set to reach target count."""
    label_counts = Counter(d["label_idx"] for d in train_data)
    augmented = []

    for cls_idx in range(NUM_CLASSES):
        cls_samples = [d for d in train_data if d["label_idx"] == cls_idx]
        n_have = len(cls_samples)
        n_need = target - n_have
        if n_need <= 0:
            continue
        cls_name = CLASSES[cls_idx]
        print(f"  {cls_name}: {n_have} -> augmenting {n_need}")
        created = 0
        while created < n_need:
            for d in cls_samples:
                if created >= n_need:
                    break
                aug_wav = augment_waveform(d["audio"].copy())
                augmented.append({
                    "audio": aug_wav,
                    "label": d["label"],
                    "label_idx": d["label_idx"],
                    "filepath": d["filepath"] + "_aug",
                    "infant_id": d["infant_id"],
                })
                created += 1

    print(f"  Total augmented: {len(augmented)}")
    return train_data + augmented


# ── Spectrogram augmentation (on-the-fly) ────────────────────────────

def spec_augment(mel_spec):
    """Apply SpecAugment: frequency and time masking."""
    spec = mel_spec.copy()
    n_mels, n_frames = spec.shape

    for _ in range(SPEC_AUG_NUM_MASKS):
        f = np.random.randint(0, SPEC_AUG_FREQ_MASK + 1)
        f0 = np.random.randint(0, max(1, n_mels - f))
        spec[f0:f0 + f, :] = 0

    for _ in range(SPEC_AUG_NUM_MASKS):
        t = np.random.randint(0, SPEC_AUG_TIME_MASK + 1)
        t0 = np.random.randint(0, max(1, n_frames - t))
        spec[:, t0:t0 + t] = 0

    return spec


# ── PyTorch Dataset ──────────────────────────────────────────────────

class CryDataset(Dataset):
    """Dataset yielding (mel_spec, domain_features, label) tuples."""

    def __init__(self, data_list, augment=False):
        self.data = data_list
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        waveform = d["audio"]
        label = d["label_idx"]

        mel = extract_mel_spectrogram(waveform)
        if self.augment:
            mel = spec_augment(mel)

        domain = extract_domain_features(waveform)

        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
        domain_tensor = torch.tensor(domain, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, domain_tensor, label_tensor


class PrecomputedDataset(Dataset):
    """Dataset with pre-extracted features for faster training."""

    def __init__(self, mel_specs, domain_feats, labels, augment=False):
        self.mel_specs = mel_specs
        self.domain_feats = domain_feats
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mel_specs[idx].copy()
        if self.augment:
            mel = spec_augment(mel)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        domain_tensor = torch.tensor(self.domain_feats[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_tensor, domain_tensor, label_tensor


def precompute_features(data_list, desc="Extracting"):
    """Pre-extract mel-spectrograms and domain features for all samples."""
    from tqdm import tqdm
    mel_specs = []
    domain_feats = []
    labels = []
    for d in tqdm(data_list, desc=desc):
        mel = extract_mel_spectrogram(d["audio"])
        dom = extract_domain_features(d["audio"])
        mel_specs.append(mel)
        domain_feats.append(dom)
        labels.append(d["label_idx"])
    return np.array(mel_specs), np.array(domain_feats), np.array(labels)


def normalize_features(train_mel, train_dom, val_mel, val_dom, test_mel, test_dom):
    """Fit scaler on train, apply to all. Returns normalized arrays."""
    mel_mean = train_mel.mean(axis=(0, 2), keepdims=True)
    mel_std = train_mel.std(axis=(0, 2), keepdims=True) + 1e-8
    train_mel = (train_mel - mel_mean) / mel_std
    val_mel = (val_mel - mel_mean) / mel_std
    test_mel = (test_mel - mel_mean) / mel_std

    dom_mean = train_dom.mean(axis=0, keepdims=True)
    dom_std = train_dom.std(axis=0, keepdims=True) + 1e-8
    train_dom = (train_dom - dom_mean) / dom_std
    val_dom = (val_dom - dom_mean) / dom_std
    test_dom = (test_dom - dom_mean) / dom_std

    return train_mel, train_dom, val_mel, val_dom, test_mel, test_dom


def make_class_balanced_sampler(labels):
    """Create a WeightedRandomSampler for class-balanced batches."""
    counts = Counter(labels.tolist() if hasattr(labels, 'tolist') else labels)
    total = sum(counts.values())
    class_weights = {c: total / n for c, n in counts.items()}
    sample_weights = [class_weights[int(l)] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


def create_dataloaders(train_mel, train_dom, train_labels,
                       val_mel, val_dom, val_labels,
                       batch_size=16, balanced=False):
    """Create train/val DataLoaders from pre-computed features."""
    train_ds = PrecomputedDataset(train_mel, train_dom, train_labels, augment=True)
    val_ds = PrecomputedDataset(val_mel, val_dom, val_labels, augment=False)

    sampler = make_class_balanced_sampler(train_labels) if balanced else None
    shuffle = not balanced

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              sampler=sampler, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
