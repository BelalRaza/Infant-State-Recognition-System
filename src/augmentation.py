"""
augmentation.py — Audio data augmentation for minority class balancing.

Generates new .wav files from existing minority-class samples using five
acoustically valid transforms.  Augmented files are saved directly into
data/raw/<class>/ so the data loader picks them up transparently.

Strategy
--------
Only minority classes are augmented.  The dominant class (hunger) is NEVER
augmented — doing so would worsen the imbalance.  Target: bring every class
to at least ``MIN_TARGET`` samples (~80) so the max ratio drops from ~54:1
to a manageable ~5:1.

Techniques
----------
1. Pitch shift   (±1, ±2 semitones)          → 4 variants
2. Time stretch  (0.9x, 1.1x)               → 2 variants
3. White noise   (SNR ≈ 25–30 dB)            → 1 variant
4. Time shift    (±0.5 s zero-padded)        → 2 variants
5. Combination   (pitch +1 semi + noise)     → 1 variant
                                     Total up to 10 per original

Only as many variants as needed are generated to reach the target count.
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

from src.config import RAW_DIR, SAMPLE_RATE, DURATION, CLASSES, CLASS_TO_IDX


# ── Configuration ─────────────────────────────────────────────────────

MIN_TARGET = 300         # minimum samples per class after augmentation
DOMINANT_CLASS = "hunger" # never augment the majority class
NOISE_FACTOR = 0.005     # Gaussian noise amplitude (~25–30 dB SNR)
COMBO_NOISE = 0.003      # lighter noise for the combination technique
GAIN_DB_RANGE = 6.0      # random gain variation in dB


# ── Individual augmentation functions ─────────────────────────────────


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by *n_steps* semitones.  Preserves duration."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Speed up (rate > 1) or slow down (rate < 1).  Preserves pitch.

    The output is resampled back to the original length so all clips
    remain the same duration for downstream processing.
    """
    y_stretched = librosa.effects.time_stretch(y=y, rate=rate)
    target_len = len(y)
    if len(y_stretched) < target_len:
        y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)))
    else:
        y_stretched = y_stretched[:target_len]
    return y_stretched


def add_white_noise(y: np.ndarray, noise_factor: float = NOISE_FACTOR) -> np.ndarray:
    """Inject low-level Gaussian noise (SNR ≈ 25–30 dB)."""
    noise = noise_factor * np.random.randn(len(y))
    return (y + noise).astype(np.float32)


def time_shift(y: np.ndarray, sr: int, shift_sec: float = 0.5) -> np.ndarray:
    """Circular-shift audio then zero-pad the gap.

    Positive *shift_sec* shifts right (silence at start);
    negative shifts left (silence at end).
    """
    shift_samples = int(sr * abs(shift_sec))
    shifted = np.roll(y, shift_samples if shift_sec > 0 else -shift_samples)
    if shift_sec > 0:
        shifted[:shift_samples] = 0.0
    else:
        shifted[-shift_samples:] = 0.0
    return shifted


def combination(y: np.ndarray, sr: int) -> np.ndarray:
    """Pitch shift +1 semitone AND add light noise."""
    y2 = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=1.0)
    y2 = y2 + COMBO_NOISE * np.random.randn(len(y2))
    return y2.astype(np.float32)


def random_gain(y: np.ndarray, db_range: float = GAIN_DB_RANGE) -> np.ndarray:
    """Apply random gain (volume variation) within ±db_range dB."""
    gain_db = np.random.uniform(-db_range, db_range)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (y * gain_linear).astype(np.float32)


def bandpass_filter(y: np.ndarray, sr: int) -> np.ndarray:
    """Simulate microphone variation via band-pass filtering.

    Randomly selects a low-cut (200-500 Hz) and high-cut (2500-3800 Hz)
    to simulate different recording conditions.
    """
    from scipy.signal import butter, sosfilt

    low_cut = np.random.uniform(200, 500)
    high_cut = np.random.uniform(2500, 3800)
    nyq = sr / 2.0
    low = low_cut / nyq
    high = min(high_cut / nyq, 0.99)
    sos = butter(4, [low, high], btype='band', output='sos')
    return sosfilt(sos, y).astype(np.float32)


def reverb_simulation(y: np.ndarray, sr: int) -> np.ndarray:
    """Simulate room impulse response (reverb) via convolution with
    a synthetic exponentially decaying noise burst."""
    rt60 = np.random.uniform(0.1, 0.4)  # seconds
    n_samples = int(sr * rt60)
    impulse = np.random.randn(n_samples)
    impulse *= np.exp(-np.linspace(0, 6, n_samples))
    impulse = impulse / np.max(np.abs(impulse))
    y_rev = np.convolve(y, impulse, mode='full')[:len(y)]
    # Mix dry/wet
    wet = np.random.uniform(0.15, 0.35)
    y_out = (1 - wet) * y + wet * y_rev
    return y_out.astype(np.float32)


# ── Augmentation registry ────────────────────────────────────────────
# Each entry: (function, kwargs_dict, suffix_tag)
# They are applied in this order until the target count is reached.

def _build_augmentation_list(sr: int):
    """Return an ordered list of (transform_fn, suffix) tuples.

    The caller passes each original waveform through these one by one,
    stopping once enough augmented files have been generated.
    """
    return [
        (lambda y: pitch_shift(y, sr, n_steps=1),    "ps+1"),
        (lambda y: pitch_shift(y, sr, n_steps=-1),   "ps-1"),
        (lambda y: pitch_shift(y, sr, n_steps=2),    "ps+2"),
        (lambda y: pitch_shift(y, sr, n_steps=-2),   "ps-2"),
        (lambda y: time_stretch(y, rate=0.9),         "ts09"),
        (lambda y: time_stretch(y, rate=1.1),         "ts11"),
        (lambda y: add_white_noise(y),                "wn"),
        (lambda y: time_shift(y, sr, shift_sec=0.5),  "sh+"),
        (lambda y: time_shift(y, sr, shift_sec=-0.5), "sh-"),
        (lambda y: combination(y, sr),                "combo"),
        (lambda y: random_gain(y),                    "gain"),
        (lambda y: bandpass_filter(y, sr),            "bp"),
        (lambda y: reverb_simulation(y, sr),          "reverb"),
        (lambda y: pitch_shift(y, sr, n_steps=0.5),   "ps+05"),
        (lambda y: pitch_shift(y, sr, n_steps=-0.5),  "ps-05"),
        (lambda y: time_stretch(y, rate=0.85),         "ts085"),
        (lambda y: time_stretch(y, rate=1.15),         "ts115"),
    ]


# ── Main augmentation driver ─────────────────────────────────────────


def augment_class(
    cls: str,
    data_dir: Path = RAW_DIR,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    target: int = MIN_TARGET,
) -> int:
    """Augment a single class to reach *target* total files.

    New .wav files are written next to the originals with an ``_aug_``
    tag in the filename so they can be identified later if needed.

    Returns the number of new files created.
    """
    cls_dir = data_dir / cls
    if not cls_dir.is_dir():
        print(f"[WARN] Directory not found: {cls_dir}")
        return 0

    # Gather only original (non-augmented) files
    originals = sorted(
        f for f in cls_dir.iterdir()
        if f.suffix.lower() == ".wav" and "_aug_" not in f.stem
    )
    n_existing = len(originals)
    n_needed = target - n_existing

    if n_needed <= 0:
        print(f"  {cls}: already has {n_existing} files (≥ {target}), skipping.")
        return 0

    print(f"  {cls}: {n_existing} originals → need {n_needed} augmented files")

    transforms = _build_augmentation_list(sr)
    target_len = int(sr * duration)
    created = 0

    # Round-robin: cycle through originals, applying transforms sequentially
    for orig_path in tqdm(originals, desc=f"  aug/{cls}", leave=False):
        if created >= n_needed:
            break

        try:
            y, _ = librosa.load(orig_path, sr=sr, duration=duration, mono=True)
        except Exception as exc:
            print(f"  [ERROR] Cannot load {orig_path.name}: {exc}")
            continue

        # Pad / truncate to fixed length
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        for transform_fn, suffix in transforms:
            if created >= n_needed:
                break
            try:
                y_aug = transform_fn(y)
                # Ensure same length
                if len(y_aug) < target_len:
                    y_aug = np.pad(y_aug, (0, target_len - len(y_aug)))
                else:
                    y_aug = y_aug[:target_len]

                out_name = f"{orig_path.stem}_aug_{suffix}.wav"
                out_path = cls_dir / out_name
                sf.write(str(out_path), y_aug, sr)
                created += 1
            except Exception as exc:
                print(f"  [ERROR] {suffix} on {orig_path.name}: {exc}")

    print(f"  {cls}: created {created} augmented files "
          f"(total now ≈ {n_existing + created})")
    return created


def run_augmentation(
    data_dir: Path = RAW_DIR,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    target: int = MIN_TARGET,
    classes: list = None,
) -> dict[str, int]:
    """Augment all minority classes.  Returns {class: n_files_created}.

    The dominant class (hunger) is explicitly excluded.
    """
    if classes is None:
        classes = CLASSES

    print("\n" + "=" * 60)
    print("  Audio Augmentation — Minority Class Balancing")
    print("=" * 60)
    print(f"  Target minimum per class : {target}")
    print(f"  Dominant class (skipped) : {DOMINANT_CLASS}")
    print(f"  Sample rate              : {sr} Hz")
    print(f"  Duration                 : {duration} s\n")

    results = {}
    for cls in classes:
        if cls == DOMINANT_CLASS:
            cls_dir = data_dir / cls
            n_existing = len([
                f for f in cls_dir.iterdir()
                if f.suffix.lower() == ".wav"
            ]) if cls_dir.is_dir() else 0
            print(f"  {cls}: {n_existing} files — DOMINANT CLASS, skipping.\n")
            results[cls] = 0
            continue
        results[cls] = augment_class(cls, data_dir, sr, duration, target)

    print("\n  Augmentation summary:")
    for cls, n in results.items():
        cls_dir = data_dir / cls
        total = len([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() == ".wav"
        ]) if cls_dir.is_dir() else 0
        print(f"    {cls:12s}  +{n:3d} aug  →  {total:4d} total")

    return results


def augment_waveforms_in_memory(
    waveforms: list[np.ndarray],
    labels: np.ndarray,
    sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    target_per_class: int = MIN_TARGET,
    dominant_class: str = DOMINANT_CLASS,
    classes: list = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Augment minority-class waveforms IN MEMORY (no disk writes).

    Returns only the newly created augmented waveforms and their labels.
    The caller concatenates these with the originals.
    """
    if classes is None:
        classes = CLASSES

    dominant_idx = CLASS_TO_IDX.get(dominant_class)
    target_len = int(sr * duration)
    transforms = _build_augmentation_list(sr)

    aug_waveforms: list[np.ndarray] = []
    aug_labels: list[int] = []

    labels_arr = np.asarray(labels)
    unique_labels = np.unique(labels_arr)

    for label_idx in unique_labels:
        if label_idx == dominant_idx:
            continue

        class_mask = labels_arr == label_idx
        class_waveforms = [waveforms[i] for i in range(len(waveforms)) if class_mask[i]]
        n_originals = len(class_waveforms)
        n_needed = target_per_class - n_originals

        if n_needed <= 0:
            continue

        cls_name = classes[label_idx] if label_idx < len(classes) else str(label_idx)
        print(f"  {cls_name}: {n_originals} train originals -> augmenting {n_needed} in memory")

        created = 0
        while created < n_needed:
            for orig in class_waveforms:
                if created >= n_needed:
                    break
                for transform_fn, suffix in transforms:
                    if created >= n_needed:
                        break
                    try:
                        y_aug = transform_fn(orig)
                        if len(y_aug) < target_len:
                            y_aug = np.pad(y_aug, (0, target_len - len(y_aug)))
                        else:
                            y_aug = y_aug[:target_len]
                        aug_waveforms.append(y_aug)
                        aug_labels.append(label_idx)
                        created += 1
                    except Exception:
                        continue

    print(f"  Total augmented samples created: {len(aug_labels)}")
    return aug_waveforms, np.array(aug_labels, dtype=np.int64)


def clean_augmented_files(data_dir: Path = RAW_DIR, classes: list = None) -> int:
    """Remove all previously generated augmented files (those with ``_aug_``
    in the filename).  Useful for re-running augmentation from scratch.

    Returns the number of files deleted.
    """
    if classes is None:
        classes = CLASSES

    deleted = 0
    for cls in classes:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() == ".wav" and "_aug_" in f.stem:
                f.unlink()
                deleted += 1

    print(f"Cleaned {deleted} augmented files.")
    return deleted
