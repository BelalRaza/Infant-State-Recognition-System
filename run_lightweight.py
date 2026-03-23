"""
run_lightweight.py — Memory-safe Phase 1 pipeline.

Processes feature extraction in batches with gc.collect() to prevent
MacBook crashes. Skips augmentation (uses existing files).
"""

import sys
import gc
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSES, RANDOM_STATE, TEST_SIZE, FEATURES_DIR, N_MFCC, SAMPLE_RATE,
    HOP_LENGTH, N_FFT, N_MELS,
)
from src.data_loader import load_dataset, get_class_distribution
from src.feature_extractor import FeatureExtractor
from src.feature_extraction import extract_mfcc, save_features
from src.gmm_classifier import GMMClassifier
from src.svm_classifier import SVMClassifier
from src.hmm_model import HMMClassifier
from src.rf_classifier import RFClassifier
from src.xgb_classifier import XGBCryClassifier
from src.ensemble_classifier import EnsembleClassifier
from src.evaluation import full_evaluation, plot_model_comparison, plot_tsne, plot_feature_importance

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def extract_features_batched(X_raw, batch_size=25):
    """Extract 411-dim features in small batches with memory cleanup."""
    extractor = FeatureExtractor()
    all_features = []
    all_mfcc_seqs = []
    feature_names = None

    n_total = len(X_raw)
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch = X_raw[start:end]
        print(f"  Batch {start//batch_size + 1}/{(n_total + batch_size - 1)//batch_size} "
              f"(samples {start+1}-{end}/{n_total})")

        for i, waveform in enumerate(batch):
            try:
                feats, names = extractor.extract_all(waveform)
                all_features.append(feats)
                if feature_names is None:
                    feature_names = names
            except Exception as exc:
                print(f"  [ERROR] Sample {start+i}: {exc}")
                all_features.append(np.zeros(411, dtype=np.float32))

            # Frame-level MFCCs for HMM
            mfcc_mat = extract_mfcc(waveform)
            all_mfcc_seqs.append(mfcc_mat.T)

        gc.collect()

    return np.array(all_features, dtype=np.float32), all_mfcc_seqs, feature_names


def main():
    print("=" * 60)
    print("  Infant Cry Classifier — Lightweight Pipeline")
    print("=" * 60)

    # ── 1. Load data (no augmentation, use existing files) ──
    print("\n[1/6] Loading audio data ...")
    X_raw, y, file_paths = load_dataset()

    if len(X_raw) == 0:
        print("No audio files found!")
        sys.exit(1)

    dist = get_class_distribution(y)
    print(f"  Total: {len(y)} samples")
    print(f"  Distribution: {dist}")

    # ── 2. Extract features in batches ──
    print("\n[2/6] Extracting 411-dim features (batched) ...")
    clip_features, mfcc_sequences, feature_names = extract_features_batched(X_raw, batch_size=25)
    print(f"  Feature shape: {clip_features.shape}")

    # Check for NaN/Inf and replace
    nan_mask = ~np.isfinite(clip_features)
    if nan_mask.any():
        n_bad = nan_mask.sum()
        print(f"  [WARN] Replacing {n_bad} NaN/Inf values with 0")
        clip_features[nan_mask] = 0.0

    # Cache features
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_DIR / "X.npy", clip_features)
    np.save(FEATURES_DIR / "y.npy", y)
    print(f"  Cached → {FEATURES_DIR}/")

    gc.collect()

    # ── 3. Train/test split ──
    print("\n[3/6] Splitting data ...")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    X_train, X_test = clip_features[train_idx], clip_features[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_seqs = [mfcc_sequences[i] for i in train_idx]
    test_seqs = [mfcc_sequences[i] for i in test_idx]

    save_features(X_train, y_train, tag="train")
    save_features(X_test, y_test, tag="test")

    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Train dist: {get_class_distribution(y_train)}")
    print(f"  Test  dist: {get_class_distribution(y_test)}")

    # ── 4. Train all models ──
    print("\n[4/6] Training models ...")
    all_metrics = {}

    # GMM
    print("\n--- GMM (Bayesian) ---")
    gmm = GMMClassifier(n_components=8, covariance_type="diag")
    gmm.fit(X_train, y_train)
    gmm.save()
    gc.collect()

    # SVM - use smaller grid to save time/memory
    print("\n--- SVM (SMOTE + OvO) ---")
    svm = SVMClassifier()
    small_grid = {
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.01],
        "svm__kernel": ["rbf"],
    }
    svm.fit_with_grid_search(X_train, y_train, param_grid=small_grid, cv=3, scoring="f1_macro")
    svm.save()
    gc.collect()

    # HMM
    print("\n--- HMM (left-right, 8 states) ---")
    hmm = HMMClassifier(n_states=8, covariance_type="diag")
    hmm.fit(train_seqs, y_train)
    hmm.save()
    gc.collect()

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RFClassifier(n_estimators=500)
    rf.fit(X_train, y_train)
    rf.save()
    gc.collect()

    # XGBoost
    print("\n--- XGBoost ---")
    xgb = XGBCryClassifier(n_estimators=300)
    xgb.fit(X_train, y_train)
    xgb.save()
    gc.collect()

    # ── 5. Evaluate all models ──
    print("\n[5/6] Evaluating ...")

    # GMM
    gmm_preds = gmm.predict(X_test)
    gmm_proba = gmm.predict_proba(X_test)
    all_metrics["GMM"] = full_evaluation(y_test, gmm_preds, "GMM", y_proba=gmm_proba)

    # SVM
    svm_preds = svm.predict(X_test)
    svm_proba = svm.predict_proba(X_test)
    all_metrics["SVM"] = full_evaluation(y_test, svm_preds, "SVM", y_proba=svm_proba)

    # HMM
    hmm_preds = hmm.predict(test_seqs)
    all_metrics["HMM"] = full_evaluation(y_test, hmm_preds, "HMM")

    # RF
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    all_metrics["RF"] = full_evaluation(y_test, rf_preds, "RF", y_proba=rf_proba)

    # XGBoost
    xgb_preds = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)
    all_metrics["XGBoost"] = full_evaluation(y_test, xgb_preds, "XGBoost", y_proba=xgb_proba)

    # Ensemble (stacking SVM + RF + XGBoost)
    print("\n--- Ensemble (Stacking) ---")
    base_models = {"SVM": svm, "RF": rf, "XGBoost": xgb}
    ensemble = EnsembleClassifier()
    ensemble.fit(X_train, y_train, base_models=base_models)
    ensemble.save()

    ens_preds = ensemble.predict(X_test)
    ens_proba = ensemble.predict_proba(X_test)
    all_metrics["Ensemble"] = full_evaluation(y_test, ens_preds, "Ensemble", y_proba=ens_proba)

    # ── 6. Visualizations ──
    print("\n[6/6] Generating plots ...")
    plot_model_comparison(all_metrics)
    plot_tsne(clip_features, y)

    rf_imp = rf.feature_importances(feature_names)
    plot_feature_importance(rf_imp, model_name="RF", top_n=20)
    xgb_imp = xgb.feature_importances(feature_names)
    plot_feature_importance(xgb_imp, model_name="XGBoost", top_n=20)

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"  {'Model':<12s} {'Acc':>7s} {'MacF1':>7s} {'WgtF1':>7s} "
          f"{'MCC':>7s} {'Kappa':>7s} {'AUC':>7s}")
    print("  " + "-" * 58)
    for name, m in all_metrics.items():
        auc = f"{m['auc_roc']:.4f}" if m.get('auc_roc') is not None else "  N/A"
        print(f"  {name:<12s} {m['accuracy']:>7.4f} {m['macro_f1']:>7.4f} "
              f"{m['weighted_f1']:>7.4f} {m['mcc']:>7.4f} {m['kappa']:>7.4f} "
              f"{auc:>7s}")

    print("\n  Top 10 features (RF):")
    for fname, imp in rf_imp[:10]:
        print(f"    {fname:<30s}  {imp:.4f}")

    print("\nDone! Results → results/metrics/ and results/plots/")


if __name__ == "__main__":
    main()
