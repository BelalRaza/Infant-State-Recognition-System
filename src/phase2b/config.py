"""Configuration for Phase 2B final-mountain experiments."""

from pathlib import Path

from src.phase2a.config import (
    CLASSES,
    CLASS_TO_IDX,
    FEATURES_DIR as PHASE2A_FEATURES_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR as PHASE2A_RESULTS_DIR,
    ROOT_DIR,
)

RESULTS_DIR = ROOT_DIR / "results" / "phase2b"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FEATURES_DIR = PHASE2A_FEATURES_DIR

RANDOM_STATE = 42
RARE_CLASS_NAMES = ["belly_pain", "burping"]
RARE_CLASS_IDXS = [CLASS_TO_IDX[name] for name in RARE_CLASS_NAMES]

DEFAULT_FEATURE_FILES = [
    "ast_embeddings.npz",
    "ast_aux_adapted_embeddings.npz",
    "ast_tta_embeddings.npz",
    "ast_aux_tta_embeddings.npz",
    "whisper_embeddings.npz",
    "handcrafted_features.npz",
    "ablation_no_aux_no_whisper_features.npz",
    "ablation_aux_no_whisper_features.npz",
    "ablation_no_aux_with_whisper_features.npz",
    "ablation_aux_with_whisper_features.npz",
    "jepa_lite_embeddings.npz",
]

SVM_PCA_COMPONENTS = [64, 128, 192, None]
SVM_C_VALUES = [1.0, 3.0, 10.0, 30.0]
SVM_GAMMAS = ["scale", 0.003, 0.01, 0.03]
SVM_USE_L2 = [False, True]

