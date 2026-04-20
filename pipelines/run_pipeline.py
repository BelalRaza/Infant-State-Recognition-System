"""
run_pipeline.py — End-to-end Phase 1 training and evaluation script.

Usage
-----
    python -m run_pipeline            # from project root
    python run_pipeline.py            # same

Steps
-----
0. (Optional) Augment minority classes  → new .wav files in data/raw/
1. Load raw audio  → waveforms + labels
2. Extract features (clip-level 411-dim + frame-level MFCC sequences)
3. Train/test split
4. Train GMM, SVM, HMM, RF, XGBoost, Ensemble
5. Evaluate all models and save results
6. Generate comparative visualizations
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSES,
    RANDOM_STATE,
    TEST_SIZE,
    FEATURES_DIR,
    N_MFCC,
)
from src.data_loader import load_dataset, get_class_distribution
from src.feature_extraction import (
    extract_features_batch,
    save_features,
    load_features,
    extract_mfcc,
)
from src.feature_extractor import FeatureExtractor
from src.gmm_classifier import GMMClassifier
from src.svm_classifier import SVMClassifier
from src.hmm_model import HMMClassifier
from src.rf_classifier import RFClassifier
from src.xgb_classifier import XGBCryClassifier
from src.ensemble_classifier import EnsembleClassifier
from src.evaluation import (
    full_evaluation,
    plot_model_comparison,
    plot_tsne,
    plot_feature_importance,
)
from src.augmentation import augment_waveforms_in_memory, clean_augmented_files


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
        "--aug-target", type=int, default=300,
        help="Minimum samples per class after augmentation (default: 300).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Infant Cry Classifier — Phase 1 Pipeline")
    print("  (leak-free: split-before-augment)")
    print("=" * 60)

    if args.clean_aug:
        print("\n[0/7] Cleaning previous augmented files …")
        clean_augmented_files()

    # ── 1. Load ONLY original audio (exclude _aug_ files) ─────────────
    print("\n[1/7] Loading original audio data (excluding augmented) …")
    X_raw, y, file_paths = load_dataset(originals_only=True)

    if len(X_raw) == 0:
        print("\nNo audio files found. Place .wav files in data/raw/<class>/")
        print("Classes expected:", CLASSES)
        sys.exit(1)

    dist = get_class_distribution(y)
    print(f"  Originals loaded: {len(y)} samples")
    print(f"  Class distribution: {dist}")

    # ── 2. Split on originals BEFORE augmentation ──────────────────────
    print("\n[2/7] Splitting ORIGINAL data (test_size={:.0%}) …".format(TEST_SIZE))

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_raw = X_raw[train_idx]
    X_test_raw = X_raw[test_idx]
    y_train_orig = y[train_idx]
    y_test = y[test_idx]

    print(f"  Originals — Train: {len(y_train_orig)} | Test: {len(y_test)}")
    print(f"  Train dist: {get_class_distribution(y_train_orig)}")
    print(f"  Test  dist: {get_class_distribution(y_test)}")

    # ── 3. Augment ONLY training set (in memory) ──────────────────────
    if args.augment:
        print("\n[3/7] Augmenting training set in memory …")
        aug_waveforms, aug_labels = augment_waveforms_in_memory(
            waveforms=list(X_train_raw),
            labels=y_train_orig,
            target_per_class=args.aug_target,
        )
        X_train_all = np.concatenate(
            [X_train_raw, np.array(aug_waveforms, dtype=np.float32)], axis=0
        )
        y_train = np.concatenate([y_train_orig, aug_labels], axis=0)
        print(f"  Training set: {len(y_train_orig)} originals + "
              f"{len(aug_labels)} augmented = {len(y_train)} total")
        print(f"  Train dist (after aug): {get_class_distribution(y_train)}")
        del aug_waveforms, aug_labels
    else:
        print("\n[3/7] Augmentation skipped (--no-augment).")
        X_train_all = X_train_raw
        y_train = y_train_orig

    # ── 4. Extract features ────────────────────────────────────────────
    print("\n[4/7] Extracting features (411-dim enhanced) …")

    extractor = FeatureExtractor()
    feature_names = None

    from tqdm import tqdm

    # Training features (originals + augmented)
    print("  Extracting training features …")
    train_clip_list = []
    train_mfcc_seqs = []
    for i, waveform in enumerate(tqdm(X_train_all, desc="Train features")):
        try:
            feats, names = extractor.extract_all(waveform)
            train_clip_list.append(feats)
            if feature_names is None:
                feature_names = names
        except Exception as exc:
            print(f"[ERROR] Train sample {i}: {exc}")
            train_clip_list.append(np.zeros(411, dtype=np.float32))
        mfcc_mat = extract_mfcc(waveform)
        train_mfcc_seqs.append(mfcc_mat.T)

    X_train = np.array(train_clip_list, dtype=np.float32)
    del train_clip_list, X_train_all

    # Test features (originals only — no augmentation)
    print("  Extracting test features …")
    test_clip_list = []
    test_mfcc_seqs = []
    for i, waveform in enumerate(tqdm(X_test_raw, desc="Test features")):
        try:
            feats, names = extractor.extract_all(waveform)
            test_clip_list.append(feats)
        except Exception as exc:
            print(f"[ERROR] Test sample {i}: {exc}")
            test_clip_list.append(np.zeros(411, dtype=np.float32))
        mfcc_mat = extract_mfcc(waveform)
        test_mfcc_seqs.append(mfcc_mat.T)

    X_test = np.array(test_clip_list, dtype=np.float32)
    del test_clip_list, X_test_raw

    print(f"  Train features: {X_train.shape} | Test features: {X_test.shape}")

    # Persist features
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    save_features(X_train, y_train, tag="train")
    save_features(X_test, y_test, tag="test")

    train_seqs = train_mfcc_seqs
    test_seqs = test_mfcc_seqs

    print(f"  Train: {len(y_train)} samples | Test: {len(y_test)} samples")

    # ── 4. Train models ───────────────────────────────────────────────
    print("\n[4/7] Training models …")
    all_metrics = {}

    # 4a — GMM (Bayesian, improved)
    print("\n--- GMM Classifier (Bayesian, class-weighted) ---")
    gmm = GMMClassifier(n_components=8, covariance_type="diag")
    gmm.fit(X_train, y_train)
    gmm.save()

    # 4b — SVM (with SMOTE + OvO + grid search)
    print("\n--- SVM Classifier (SMOTE + balanced + OvO) ---")
    svm = SVMClassifier()
    svm.fit_with_grid_search(X_train, y_train, cv=5, scoring="f1_macro")
    svm.save()

    # 4c — HMM (left-right topology, 8 states)
    print("\n--- HMM Classifier (left-right, 8 states) ---")
    hmm = HMMClassifier(n_states=8, covariance_type="diag")
    hmm.fit(train_seqs, y_train)
    hmm.save()

    # 4d — Random Forest (NEW)
    print("\n--- Random Forest Classifier ---")
    rf = RFClassifier(n_estimators=500)
    rf.fit(X_train, y_train)
    rf.save()

    # 4e — XGBoost (NEW)
    print("\n--- XGBoost Classifier ---")
    xgb = XGBCryClassifier(n_estimators=300)
    xgb.fit(X_train, y_train)
    xgb.save()

    # ── 5. Evaluate all models ────────────────────────────────────────
    print("\n[5/7] Evaluating models …")

    # GMM
    gmm_preds = gmm.predict(X_test)
    gmm_proba = gmm.predict_proba(X_test)
    all_metrics["GMM"] = full_evaluation(
        y_test, gmm_preds, model_name="GMM", y_proba=gmm_proba,
    )

    # SVM
    svm_preds = svm.predict(X_test)
    svm_proba = svm.predict_proba(X_test)
    all_metrics["SVM"] = full_evaluation(
        y_test, svm_preds, model_name="SVM", y_proba=svm_proba,
    )

    # HMM
    hmm_preds = hmm.predict(test_seqs)
    all_metrics["HMM"] = full_evaluation(
        y_test, hmm_preds, model_name="HMM",
    )

    # Random Forest
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    all_metrics["RF"] = full_evaluation(
        y_test, rf_preds, model_name="RF", y_proba=rf_proba,
    )

    # XGBoost
    xgb_preds = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)
    all_metrics["XGBoost"] = full_evaluation(
        y_test, xgb_preds, model_name="XGBoost", y_proba=xgb_proba,
    )

    # 5f — Ensemble (stacking SVM + RF + XGBoost)
    print("\n--- Ensemble Classifier (Stacking) ---")
    base_models = {"SVM": svm, "RF": rf, "XGBoost": xgb}
    ensemble = EnsembleClassifier()
    ensemble.fit(X_train, y_train, base_models=base_models)
    ensemble.save()

    ens_preds = ensemble.predict(X_test)
    ens_proba = ensemble.predict_proba(X_test)
    all_metrics["Ensemble"] = full_evaluation(
        y_test, ens_preds, model_name="Ensemble", y_proba=ens_proba,
    )

    # ── 6. Comparative visualizations ─────────────────────────────────
    print("\n[6/7] Generating comparative visualizations …")

    # Model comparison bar chart
    plot_model_comparison(all_metrics)

    # t-SNE visualization (on test set only for honest representation)
    all_features = np.concatenate([X_train, X_test], axis=0)
    all_labels = np.concatenate([y_train, y_test], axis=0)
    plot_tsne(all_features, all_labels)

    # Feature importance from RF and XGBoost
    rf_importances = rf.feature_importances(feature_names)
    plot_feature_importance(rf_importances, model_name="RF", top_n=20)

    xgb_importances = xgb.feature_importances(feature_names)
    plot_feature_importance(xgb_importances, model_name="XGBoost", top_n=20)

    # ── 7. Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"\n  {'Model':<12s} {'Acc':>7s} {'MacF1':>7s} {'WgtF1':>7s} "
          f"{'MCC':>7s} {'Kappa':>7s} {'AUC':>7s}")
    print("  " + "-" * 55)
    for name, m in all_metrics.items():
        auc = f"{m['auc_roc']:.4f}" if m.get('auc_roc') is not None else "  N/A"
        print(f"  {name:<12s} {m['accuracy']:>7.4f} {m['macro_f1']:>7.4f} "
              f"{m['weighted_f1']:>7.4f} {m['mcc']:>7.4f} {m['kappa']:>7.4f} "
              f"{auc:>7s}")

    print("\n  Top 10 most important features (RF):")
    for fname, imp in rf_importances[:10]:
        print(f"    {fname:<30s}  {imp:.4f}")

    print("\nDone. Results saved to results/metrics/ and results/plots/.")


if __name__ == "__main__":
    main()
