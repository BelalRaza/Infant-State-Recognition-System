"""
run_pipeline.py — End-to-end Phase 1 training and evaluation script.

Usage
-----
    python -m run_pipeline            # from infant-cry-classifier/
    python run_pipeline.py            # same, if working dir is project root

Steps
-----
0. (Optional) Augment minority classes  → new .wav files in data/raw/
1. Load raw audio  → waveforms + labels
2. Extract features (clip-level + frame-level)
3. Train/test split
4. Train GMM, SVM, (optional) HMM
5. Evaluate all models and save results
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSES,
    RANDOM_STATE,
    TEST_SIZE,
    FEATURES_DIR,
)
from src.data_loader import load_dataset, get_class_distribution
from src.feature_extraction import (
    extract_features_batch,
    save_features,
    load_features,
)
from src.gmm_classifier import GMMClassifier
from src.svm_classifier import SVMClassifier
from src.hmm_model import HMMClassifier
from src.evaluation import full_evaluation
from src.augmentation import run_augmentation, clean_augmented_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infant Cry Classifier — Phase 1 Pipeline",
    )
    parser.add_argument(
        "--augment", action="store_true", default=True,
        help="Run audio augmentation on minority classes before training "
             "(default: True).",
    )
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false",
        help="Skip augmentation and use raw data only.",
    )
    parser.add_argument(
        "--clean-aug", action="store_true",
        help="Delete all previously generated augmented files and re-augment.",
    )
    parser.add_argument(
        "--aug-target", type=int, default=80,
        help="Minimum samples per class after augmentation (default: 80).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Infant Cry Classifier — Phase 1 Pipeline")
    print("=" * 60)

    # ── 0. Augmentation (Layer 1) ─────────────────────────────────────
    if args.clean_aug:
        print("\n[0/6] Cleaning previous augmented files …")
        clean_augmented_files()

    if args.augment:
        print("\n[0/6] Augmenting minority classes …")
        run_augmentation(target=args.aug_target)
    else:
        print("\n[0/6] Augmentation skipped (--no-augment).")

    # ── 1. Load data ───────────────────────────────────────────────────
    print("\n[1/6] Loading audio data …")
    X_raw, y, file_paths = load_dataset()

    if len(X_raw) == 0:
        print("\nNo audio files found. Place .wav files in data/raw/<class>/")
        print("Classes expected:", CLASSES)
        sys.exit(1)

    dist = get_class_distribution(y)
    print("Class distribution:", dist)

    # ── 2. Extract features ────────────────────────────────────────────
    print("\n[2/6] Extracting features …")
    clip_features, mfcc_sequences = extract_features_batch(
        X_raw, return_frame_level=True,
    )
    print(f"Clip-level feature shape: {clip_features.shape}")

    # ── 3. Train / test split ──────────────────────────────────────────
    print("\n[3/6] Splitting data (test_size={:.0%}) …".format(TEST_SIZE))

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test = clip_features[train_idx], clip_features[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_seqs = [mfcc_sequences[i] for i in train_idx]
    test_seqs = [mfcc_sequences[i] for i in test_idx]

    save_features(X_train, y_train, tag="train")
    save_features(X_test, y_test, tag="test")

    print(f"  Train: {len(y_train)} samples | Test: {len(y_test)} samples")
    print(f"  Train distribution: {get_class_distribution(y_train)}")
    print(f"  Test  distribution: {get_class_distribution(y_test)}")

    # ── 4. Train models ───────────────────────────────────────────────
    print("\n[4/6] Training models …")

    # 4a — GMM (with class-weighted log-priors — Layer 2)
    print("\n--- GMM Classifier (class-weighted) ---")
    gmm = GMMClassifier(n_components=4, covariance_type="diag")
    gmm.fit(X_train, y_train)
    gmm.save()

    # 4b — SVM (with grid search + class_weight='balanced' — Layer 2)
    print("\n--- SVM Classifier (balanced class weights) ---")
    svm = SVMClassifier()
    svm.fit_with_grid_search(X_train, y_train, cv=5, scoring="f1_macro")
    svm.save()

    # 4c — HMM (exploratory / bonus)
    print("\n--- HMM Classifier (exploratory) ---")
    hmm = HMMClassifier(n_states=5, covariance_type="diag")
    hmm.fit(train_seqs, y_train)
    hmm.save()

    # ── 5. Evaluate ───────────────────────────────────────────────────
    print("\n[5/6] Evaluating models …")

    gmm_preds = gmm.predict(X_test)
    gmm_metrics = full_evaluation(y_test, gmm_preds, model_name="GMM")

    svm_preds = svm.predict(X_test)
    svm_metrics = full_evaluation(y_test, svm_preds, model_name="SVM")

    hmm_preds = hmm.predict(test_seqs)
    hmm_metrics = full_evaluation(y_test, hmm_preds, model_name="HMM")

    # ── 6. Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, m in [("GMM", gmm_metrics), ("SVM", svm_metrics), ("HMM", hmm_metrics)]:
        print(f"  {name:4s}  Acc={m['accuracy']:.4f}  "
              f"F1_macro={m['macro_f1']:.4f}  "
              f"F1_weighted={m['weighted_f1']:.4f}")

    print("\nDone. Results saved to results/metrics/ and results/plots/.")


if __name__ == "__main__":
    main()
