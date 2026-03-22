"""
model_a_svm.py — Train and evaluate the SVM classifier (Phase 1, Model A).

Standalone entry-point that:
  0. Augments minority classes (same as run_pipeline.py)
  1. Loads audio and extracts features (or uses cache if still valid)
  2. Trains an RBF-kernel SVM with grid-searched hyperparameters and
     balanced class weights
  3. Evaluates on the held-out test set
  4. Saves metrics + confusion matrix to results/

Usage
-----
    python -m src.model_a_svm          # from project root
    python src/model_a_svm.py          # same effect
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSES, RANDOM_STATE, TEST_SIZE, FEATURES_DIR, RAW_DIR,
)
from src.data_loader import load_dataset, get_class_distribution
from src.feature_extractor import FeatureExtractor
from src.svm_classifier import SVMClassifier
from src.evaluation import full_evaluation
from src.augmentation import run_augmentation

from sklearn.model_selection import train_test_split


# ── Helpers ───────────────────────────────────────────────────────────

def _count_wav_files(data_dir: Path = RAW_DIR) -> int:
    """Count total .wav files across all class sub-directories."""
    total = 0
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if cls_dir.is_dir():
            total += sum(1 for f in cls_dir.iterdir()
                         if f.suffix.lower() == ".wav")
    return total


def _cache_is_valid(X_path: Path, y_path: Path, expected_n: int) -> bool:
    """Return True only if cached features exist AND match the current
    number of .wav files on disk (i.e. post-augmentation count)."""
    if not (X_path.exists() and y_path.exists()):
        return False
    try:
        cached_n = np.load(X_path, mmap_mode="r").shape[0]
        return cached_n == expected_n
    except Exception:
        return False


def main() -> None:
    print("=" * 60)
    print("  Phase 1 — SVM Classifier")
    print("=" * 60)

    # ── 0. Augment minority classes ──────────────────────────────
    print("\n[0/4] Augmenting minority classes …")
    run_augmentation()

    # ── 1. Load / extract features ───────────────────────────────
    X_path = FEATURES_DIR / "X.npy"
    y_path = FEATURES_DIR / "y.npy"
    n_wav = _count_wav_files()

    if _cache_is_valid(X_path, y_path, n_wav):
        print(f"\n[1/4] Loading cached features ({n_wav} samples) …")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        print(f"\n[1/4] Cache stale or missing (disk: {n_wav} wav files) "
              f"— re-extracting features …")
        dataset_list, _ = load_dataset(return_list=True)
        extractor = FeatureExtractor()
        X, y, _ = extractor.extract_and_save_dataset(dataset_list)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Class distribution: {get_class_distribution(y)}")

    # ── 2. Train / test split ────────────────────────────────────
    print("\n[2/4] Splitting data (test_size={:.0%}) …".format(TEST_SIZE))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")

    # ── 3. Train SVM with grid search ────────────────────────────
    print("\n[3/4] Training SVM classifier (grid search) …")
    svm = SVMClassifier()
    svm.fit_with_grid_search(X_train, y_train, cv=5, scoring="f1_macro")
    svm.save()

    # ── 4. Evaluate ──────────────────────────────────────────────
    print("\n[4/4] Evaluation")

    # Inference timing
    start = time.perf_counter()
    preds = svm.predict(X_test)
    elapsed = (time.perf_counter() - start) * 1000  # ms total
    ms_per_sample = elapsed / len(X_test)

    metrics = full_evaluation(y_test, preds, model_name="SVM")

    print(f"\n  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:  {metrics['weighted_f1']:.4f}")
    print(f"  Inference:    {ms_per_sample:.2f} ms/sample")

    if svm.best_params:
        print(f"  Best params:  {svm.best_params}")

    print("\nDone. Results saved to results/.")


if __name__ == "__main__":
    main()
