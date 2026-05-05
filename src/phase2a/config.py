"""Configuration for Phase 2A pretrained-audio experiments."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_LAKE_DIR = ROOT_DIR / "data_lake"

SUPERVISED_MANIFEST = DATA_LAKE_DIR / "manifests" / "cause5_authentic_manifest_v2.csv"
UNIFIED_MANIFEST = DATA_LAKE_DIR / "manifests" / "unified_manifest_v2.csv"

FEATURES_DIR = DATA_LAKE_DIR / "features" / "phase2a"
HF_CACHE_DIR = DATA_LAKE_DIR / "cache" / "huggingface"
RESULTS_DIR = ROOT_DIR / "results" / "phase2a"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

CLASSES = ["hunger", "belly_pain", "burping", "discomfort", "tiredness"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

SAMPLE_RATE = 16_000
MAX_DURATION_SEC = 10.0
RANDOM_STATE = 42

TEST_SIZE = 0.15
VAL_SIZE = 0.15

AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_MODEL_NAME_OR_PATH = AST_MODEL_NAME
WHISPER_MODEL_NAME = "openai/whisper-tiny"

DEFAULT_EMBEDDING_MODELS = ["ast"]
SUPPORTED_EMBEDDING_MODELS = ["ast", "whisper"]
FEATURE_CHECKPOINT_EVERY = 25

PCA_COMPONENTS = [32, 64, 128]
ABSTAIN_THRESHOLDS = [0.40, 0.50, 0.60, 0.70]

AUX_RESULTS_DIR = RESULTS_DIR / "auxiliary_adaptation"
AUX_APPROVED_SPLITS = ["aux_pretrain", "external_train", "train"]
AUX_EXCLUDED_LABELS = ["unknown_or_uncertain"]

