"""
config.py — Central configuration for the infant-cry-classifier project.

All hyper-parameters, directory paths, and class labels live here so that
every other module imports from a single source of truth.
"""

from pathlib import Path

# ── Directory layout ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent          # infant-cry-classifier/
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = ROOT_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORT_DIR = ROOT_DIR / "report"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# ── Audio parameters ───────────────────────────────────────────────────
SAMPLE_RATE = 8000           # Hz — native rate of the dataset (no upsampling)
DURATION = 7                 # seconds — clips are padded or truncated to this
N_SAMPLES = SAMPLE_RATE * DURATION   # total samples per clip (56 000)

# ── Feature-extraction parameters ──────────────────────────────────────
N_MFCC = 40                 # number of Mel-frequency cepstral coefficients
HOP_LENGTH = 160             # STFT hop in samples (~20 ms at 8 kHz)
N_FFT = 512                  # FFT window size in samples (~64 ms at 8 kHz)
N_MELS = 64                  # Mel filter-bank size (appropriate for 4 kHz Nyquist)

# ── Class labels ───────────────────────────────────────────────────────
CLASSES = [
    "hunger",
    "belly_pain",
    "burping",
    "discomfort",
    "tiredness",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# ── Reproducibility ───────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2              # 80/20 train-test split
