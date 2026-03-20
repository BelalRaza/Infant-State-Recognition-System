"""
model_a_gmm.py — Train and evaluate the GMM classifier (Phase 1, Model A).

Standalone entry-point that loads pre-extracted features, trains a
class-weighted GMM (one mixture per class), evaluates on the held-out
test set, and saves metrics + confusion matrix to results/.

Usage
-----
    python -m src.model_a_gmm          # from project root
    python src/model_a_gmm.py          # same effect
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSES, RANDOM_STATE, TEST_SIZE, FEATURES_DIR, PLOTS_DIR, METRICS_DIR,
)
from src.data_loader import load_dataset, get_class_distribution
from src.feature_extractor import FeatureExtractor
from src.gmm_classifier import GMMClassifier
from src.evaluation import full_evaluation

from sklearn.model_selection import train_test_split


def main() -> None:
    print("=" * 60)
    print("  Phase 1 — GMM Classifier")
    print("=" * 60)

    # ── Load features ─────────────────────────────────────────────
    X_path = FEATURES_DIR / "X.npy"
    y_path = FEATURES_DIR / "y.npy"

    if X_path.exists() and y_path.exists():
        print("\n[1/3] Loading cached features …")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        print("\n[1/3] Extracting features (no cache found) …")
        dataset_list, _ = load_dataset(return_list=True)
        extractor = FeatureExtractor()
        X, y, _ = extractor.extract_and_save_dataset(dataset_list)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Class distribution: {get_class_distribution(y)}")

    # ── Train / test split ────────────────────────────────────────
    print("\n[2/3] Splitting data (test_size={:.0%}) …".format(TEST_SIZE))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")

    # ── Train GMM ─────────────────────────────────────────────────
    print("\n[3/3] Training GMM classifier …")
    gmm = GMMClassifier(n_components=4, covariance_type="diag")
    gmm.fit(X_train, y_train)
    gmm.save()

    # ── Evaluate ──────────────────────────────────────────────────
    print("\n── Evaluation ──")

    # Inference timing
    start = time.perf_counter()
    preds = gmm.predict(X_test)
    elapsed = (time.perf_counter() - start) * 1000  # ms total
    ms_per_sample = elapsed / len(X_test)

    metrics = full_evaluation(y_test, preds, model_name="GMM")

    print(f"\n  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:  {metrics['weighted_f1']:.4f}")
    print(f"  Inference:    {ms_per_sample:.2f} ms/sample")

    print("\nDone. Results saved to results/.")


if __name__ == "__main__":
    main()
