"""Phase 2 configuration — all hyperparameters in one place."""

from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = ROOT_DIR / "features"
MODELS_DIR = ROOT_DIR / "models" / "phase2"
RESULTS_DIR = ROOT_DIR / "results" / "phase2"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"

# ── Audio ────────────────────────────────────────────────────────────
SAMPLE_RATE = 8000
DURATION = 7
N_SAMPLES = SAMPLE_RATE * DURATION

# ── Classes ──────────────────────────────────────────────────────────
CLASSES = ["hunger", "belly_pain", "burping", "discomfort", "tiredness"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# ── Class counts (originals) ────────────────────────────────────────
CLASS_COUNTS = [382, 16, 8, 27, 24]

# ── Mel-spectrogram (optimized for 8 kHz) ────────────────────────────
MEL_N_FFT = 512
MEL_HOP = 128
MEL_N_MELS = 64
MEL_FMIN = 50
MEL_FMAX = 4000

# ── Domain features ─────────────────────────────────────────────────
DOMAIN_FEAT_DIM = 32
N_MFCC_DOMAIN = 14  # extract 14 so we can use indices 1-13 (skip c0 energy)

# ── Split ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# ── Augmentation ─────────────────────────────────────────────────────
AUG_TARGET_PER_CLASS = 384
AUG_NOISE_AMP = (0.005, 0.015)
AUG_TIME_SHIFT_SEC = 0.5
AUG_STRETCH_RANGE = (0.9, 1.1)
AUG_SNR_RANGE = (10, 20)
SPEC_AUG_FREQ_MASK = 10
SPEC_AUG_TIME_MASK = 20
SPEC_AUG_NUM_MASKS = 2
MIXUP_ALPHA = 0.2

# ── Student model (~40K params) ──────────────────────────────────────
STUDENT_CNN_CHANNELS = [16, 32]
STUDENT_LSTM_HIDDEN = 20
STUDENT_ATTN_DIM = 20
STUDENT_FUSION_DIM = 40
STUDENT_FEAT_HIDDEN = 32

# ── Teacher model (~200K params) ─────────────────────────────────────
TEACHER_CNN_CHANNELS = [32, 64]
TEACHER_LSTM_HIDDEN = 64
TEACHER_ATTN_DIM = 32
TEACHER_FUSION_DIM = 128
TEACHER_FEAT_HIDDEN = 64

# ── Training ─────────────────────────────────────────────────────────
BATCH_SIZE = 16
MAX_EPOCHS_STUDENT = 100
MAX_EPOCHS_TEACHER = 150
LR = 1e-3
LR_MIN = 1e-6
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-5
LR_PATIENCE = 4
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 8
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3
SE_REDUCTION = 8

# ── LDAM + DRW ───────────────────────────────────────────────────────
DRW_SWITCH_EPOCH = 60

# ── Decoupled training (cRT) ────────────────────────────────────────
CRT_EPOCHS = 30

# ── Hierarchical ─────────────────────────────────────────────────────
HIER_THRESHOLD = 0.3

# ── Knowledge distillation ───────────────────────────────────────────
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.7

# ── Ablation variants ───────────────────────────────────────────────
ABLATION_NAMES = [
    "full_model",
    "no_attention",
    "no_bilstm",
    "spec_only",
    "feat_only",
    "cnn_only",
    "ce_loss",
    "no_augmentation",
    "flat_5class",
    "teacher_200k",
]
