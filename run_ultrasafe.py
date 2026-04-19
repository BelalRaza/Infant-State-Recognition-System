"""
run_ultrasafe.py — Ultra memory-safe pipeline for MacBook.

KEY CHANGES vs run_lightweight.py:
- Skips pyin pitch tracking (the #1 memory killer)
- Processes ONE file at a time with gc.collect() after each
- Uses only 100 RF trees and 100 XGBoost rounds
- Smaller SVM grid (single combo)
- Saves progress after each stage so crashes don't lose work
- Reduces feature dim from 411 to 404 (fills pitch with zeros)
"""

import sys
import gc
import os
import json
import numpy as np
from pathlib import Path

# Limit threads to prevent memory explosion
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

from src.config import (
    CLASSES, RANDOM_STATE, TEST_SIZE, FEATURES_DIR, MODELS_DIR,
    RESULTS_DIR, PLOTS_DIR, METRICS_DIR,
    N_MFCC, SAMPLE_RATE, HOP_LENGTH, N_FFT, N_MELS,
    CQT_N_BINS, CQT_BINS_PER_OCTAVE, N_CQCC, N_CONTRAST_BANDS,
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def extract_single_safe(audio, sr=SAMPLE_RATE):
    """Extract features for ONE audio clip without pyin (memory safe).

    Returns 411-dim vector (pitch features filled with estimates based on spectral centroid).
    """
    import librosa
    from scipy.fft import dct

    hop = HOP_LENGTH

    # 1. MFCC block (240 dims)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                 hop_length=hop, n_fft=N_FFT, n_mels=N_MELS)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_vec = np.concatenate([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1),
    ]).astype(np.float32)
    del delta, delta2

    # 2. CQCC block (120 dims)
    # Use fewer bins to stay under Nyquist at 8kHz (max freq < 4000Hz)
    cqt_bins = min(CQT_N_BINS, 60)  # 5 octaves x 12 instead of 7x12
    C = np.abs(librosa.cqt(y=audio, sr=sr, hop_length=hop,
                            n_bins=cqt_bins, bins_per_octave=CQT_BINS_PER_OCTAVE))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    cqcc = dct(C_db, type=2, axis=0, norm='ortho')[:N_CQCC]
    d_cqcc = librosa.feature.delta(cqcc, order=1)
    d2_cqcc = librosa.feature.delta(cqcc, order=2)
    cqcc_vec = np.concatenate([
        np.mean(cqcc, axis=1), np.std(cqcc, axis=1),
        np.mean(d_cqcc, axis=1), np.std(d_cqcc, axis=1),
        np.mean(d2_cqcc, axis=1), np.std(d2_cqcc, axis=1),
    ]).astype(np.float32)
    del C, C_db, cqcc, d_cqcc, d2_cqcc

    # 3. Pitch features (7 dims) — ESTIMATE from spectral centroid instead of pyin
    centroid_frames = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop)
    cent_vals = centroid_frames[0]
    # Rough F0 proxy: spectral centroid / 3 (heuristic for harmonic content)
    f0_est = cent_vals / 3.0
    f0_est = f0_est[f0_est > 80]  # filter unreasonable values
    if len(f0_est) > 1:
        pitch_vec = np.array([
            np.mean(f0_est), np.std(f0_est), np.max(f0_est),
            np.min(f0_est), np.max(f0_est) - np.min(f0_est),
            len(f0_est) / len(cent_vals),  # "voiced fraction"
            np.polyfit(range(len(f0_est)), f0_est, 1)[0] if len(f0_est) > 1 else 0,
        ], dtype=np.float32)
    else:
        pitch_vec = np.zeros(7, dtype=np.float32)
    del centroid_frames, cent_vals

    # 4. Spectral contrast (14 dims) — reduce bands for low sample rate
    n_bands = min(N_CONTRAST_BANDS, 4)  # safe for 8kHz Nyquist
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=n_bands, fmin=100, hop_length=hop)
    contrast_stats = np.concatenate([np.mean(contrast, axis=1), np.std(contrast, axis=1)]).astype(np.float32)
    # Pad to 14 dims to keep consistent feature vector size
    contrast_vec = np.zeros(14, dtype=np.float32)
    contrast_vec[:len(contrast_stats)] = contrast_stats
    del contrast

    # 5. Chroma (24 dims)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop)
    chroma_vec = np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)]).astype(np.float32)
    del chroma

    # 6. Statistical features (6 dims)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio, hop_length=hop)))
    centroid_val = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop, roll_percent=0.85)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop)))
    rms = float(np.mean(librosa.feature.rms(y=audio, hop_length=hop)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio, hop_length=hop)))
    stat_vec = np.array([zcr, centroid_val, rolloff, bandwidth, rms, flatness], dtype=np.float32)

    # Also get MFCC matrix for HMM (13 coefficients only)
    mfcc_hmm = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop, n_fft=N_FFT, n_mels=N_MELS)
    mfcc_seq = mfcc_hmm.T  # (T, 13)
    del mfcc_hmm

    features = np.concatenate([mfcc_vec, cqcc_vec, pitch_vec, contrast_vec, chroma_vec, stat_vec])
    gc.collect()
    return features, mfcc_seq


def load_files_list():
    """Get file paths and labels for ORIGINAL files only (exclude augmented)."""
    from src.config import RAW_DIR
    files = []
    audio_ext = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    for idx, cls in enumerate(CLASSES):
        cls_dir = RAW_DIR / cls
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in audio_ext and "_aug_" not in f.stem:
                files.append((str(f), idx, cls))
    return files


def main():
    import librosa
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("  Infant Cry Classifier — ULTRA-SAFE Pipeline")
    print("  (leak-free: split-before-augment, single-file processing)")
    print("=" * 60)

    from src.augmentation import augment_waveforms_in_memory

    # Invalidate old cached features (built from leaky pipeline)
    feat_cache = CHECKPOINT_DIR / "features_411.npz"
    if feat_cache.exists():
        print(f"\n[WARN] Removing stale feature cache: {feat_cache}")
        feat_cache.unlink()

    # ── 1. Get ORIGINAL file list (no _aug_ files) ──
    print("\n[1/6] Scanning original audio files...")
    file_list = load_files_list()
    print(f"  Found {len(file_list)} original files")

    from collections import Counter
    dist = Counter(cls for _, _, cls in file_list)
    for cls in CLASSES:
        print(f"    {cls}: {dist.get(cls, 0)}")

    # ── 2. Split file list BEFORE extraction ──
    print(f"\n[2/6] Splitting originals...")
    file_labels = np.array([label_idx for _, label_idx, _ in file_list])
    file_indices = np.arange(len(file_list))
    train_file_idx, test_file_idx = train_test_split(
        file_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=file_labels,
    )

    train_files = [file_list[i] for i in train_file_idx]
    test_files = [file_list[i] for i in test_file_idx]
    print(f"  Train: {len(train_files)} | Test: {len(test_files)} originals")

    # ── 3. Load train originals, augment in memory, extract features ──
    print(f"\n[3/6] Processing training set (load + augment + extract)...")

    # Load training waveforms
    train_waveforms = []
    train_labels_orig = []
    target_len = SAMPLE_RATE * 7
    for fpath, label_idx, cls in train_files:
        try:
            audio, _ = librosa.load(fpath, sr=SAMPLE_RATE, duration=7, mono=True)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            train_waveforms.append(audio.astype(np.float32))
            train_labels_orig.append(label_idx)
        except Exception as e:
            print(f"    [ERROR] {Path(fpath).name}: {e}")
    gc.collect()

    y_train_orig = np.array(train_labels_orig, dtype=np.int64)

    # Augment training set in memory
    print("  Augmenting training set in memory...")
    aug_waveforms, aug_labels = augment_waveforms_in_memory(
        waveforms=train_waveforms,
        labels=y_train_orig,
        target_per_class=300,
    )
    all_train_waveforms = train_waveforms + aug_waveforms
    y_train = np.concatenate([y_train_orig, aug_labels], axis=0)
    print(f"  Training: {len(train_waveforms)} originals + {len(aug_waveforms)} aug = {len(y_train)}")
    del train_waveforms, aug_waveforms, aug_labels
    gc.collect()

    # Extract training features one at a time
    print(f"  Extracting training features ({len(all_train_waveforms)} samples)...")
    train_features = []
    train_mfcc_seqs = []
    for i, audio in enumerate(all_train_waveforms):
        try:
            feats, mfcc_seq = extract_single_safe(audio)
            train_features.append(feats)
            train_mfcc_seqs.append(mfcc_seq)
        except Exception as e:
            print(f"    [ERROR] Train sample {i}: {e}")
            train_features.append(np.zeros(411, dtype=np.float32))
            train_mfcc_seqs.append(np.zeros((10, 13), dtype=np.float32))
        del audio
        gc.collect()
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(all_train_waveforms)}] processed")

    del all_train_waveforms
    gc.collect()

    X_train = np.array(train_features, dtype=np.float32)
    del train_features

    # Fix NaN/Inf
    nan_mask = ~np.isfinite(X_train)
    if nan_mask.any():
        print(f"  [WARN] Replacing {nan_mask.sum()} NaN/Inf with 0 in train")
        X_train[nan_mask] = 0.0

    # ── 4. Extract test features (originals only) ──
    print(f"\n[4/6] Extracting test features ({len(test_files)} originals)...")
    test_features = []
    test_mfcc_seqs = []
    test_labels = []
    for i, (fpath, label_idx, cls) in enumerate(test_files):
        try:
            audio, _ = librosa.load(fpath, sr=SAMPLE_RATE, duration=7, mono=True)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]
            audio = audio.astype(np.float32)

            feats, mfcc_seq = extract_single_safe(audio)
            test_features.append(feats)
            test_mfcc_seqs.append(mfcc_seq)
            test_labels.append(label_idx)

            del audio
            gc.collect()
        except Exception as e:
            print(f"    [ERROR] {Path(fpath).name}: {e}")
            test_features.append(np.zeros(411, dtype=np.float32))
            test_mfcc_seqs.append(np.zeros((10, 13), dtype=np.float32))
            test_labels.append(label_idx)

    X_test = np.array(test_features, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int64)
    del test_features, test_labels

    nan_mask = ~np.isfinite(X_test)
    if nan_mask.any():
        print(f"  [WARN] Replacing {nan_mask.sum()} NaN/Inf with 0 in test")
        X_test[nan_mask] = 0.0

    gc.collect()

    train_seqs = train_mfcc_seqs
    test_seqs = test_mfcc_seqs

    from src.feature_extraction import save_features
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    save_features(X_train, y_train, tag="train")
    save_features(X_test, y_test, tag="test")

    train_dist = Counter(y_train)
    test_dist = Counter(y_test)
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Train dist: { {CLASSES[k]: v for k, v in sorted(train_dist.items())} }")
    print(f"  Test  dist: { {CLASSES[k]: v for k, v in sorted(test_dist.items())} }")

    # ── 4. Train models ONE AT A TIME ──
    print(f"\n[4/6] Training models...")
    all_metrics = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    from src.evaluation import full_evaluation, plot_model_comparison, plot_tsne, plot_feature_importance

    # --- GMM ---
    print("\n--- GMM (Bayesian) ---")
    try:
        from src.gmm_classifier import GMMClassifier
        gmm = GMMClassifier(n_components=8, covariance_type="diag")
        gmm.fit(X_train, y_train)
        gmm.save()
        gmm_preds = gmm.predict(X_test)
        gmm_proba = gmm.predict_proba(X_test)
        all_metrics["GMM"] = full_evaluation(y_test, gmm_preds, "GMM", y_proba=gmm_proba)
        del gmm, gmm_preds, gmm_proba
    except Exception as e:
        print(f"  GMM FAILED: {e}")
    gc.collect()

    # --- SVM (no grid search — just fit with good defaults) ---
    print("\n--- SVM (SMOTE + OvO) ---")
    try:
        from src.svm_classifier import SVMClassifier
        svm = SVMClassifier(C=10, gamma="scale", kernel="rbf")
        # Skip grid search to save memory — just fit directly
        svm.fit(X_train, y_train)
        svm.save()
        svm_preds = svm.predict(X_test)
        svm_proba = svm.predict_proba(X_test)
        all_metrics["SVM"] = full_evaluation(y_test, svm_preds, "SVM", y_proba=svm_proba)
        del svm_preds, svm_proba
    except Exception as e:
        print(f"  SVM FAILED: {e}")
        svm = None
    gc.collect()

    # --- HMM --- SKIPPED (causes OOM on MacBook)
    print("\n--- HMM --- SKIPPED (memory safety)")
    del train_seqs, test_seqs
    gc.collect()

    # --- Random Forest (reduced trees) ---
    print("\n--- Random Forest ---")
    try:
        from src.rf_classifier import RFClassifier
        rf = RFClassifier(n_estimators=200)
        rf.fit(X_train, y_train)
        rf.save()
        rf_preds = rf.predict(X_test)
        rf_proba = rf.predict_proba(X_test)
        all_metrics["RF"] = full_evaluation(y_test, rf_preds, "RF", y_proba=rf_proba)
        del rf_preds, rf_proba
    except Exception as e:
        print(f"  RF FAILED: {e}")
        rf = None
    gc.collect()

    # --- XGBoost (reduced rounds) ---
    print("\n--- XGBoost ---")
    try:
        from src.xgb_classifier import XGBCryClassifier
        xgb = XGBCryClassifier(n_estimators=150)
        xgb.fit(X_train, y_train)
        xgb.save()
        xgb_preds = xgb.predict(X_test)
        xgb_proba = xgb.predict_proba(X_test)
        all_metrics["XGBoost"] = full_evaluation(y_test, xgb_preds, "XGBoost", y_proba=xgb_proba)
        del xgb_preds, xgb_proba
    except Exception as e:
        print(f"  XGBoost FAILED: {e}")
        xgb = None
    gc.collect()

    # --- Ensemble ---
    print("\n--- Ensemble (Stacking) ---")
    try:
        from src.ensemble_classifier import EnsembleClassifier
        base_models = {}
        if svm is not None:
            base_models["SVM"] = svm
        if rf is not None:
            base_models["RF"] = rf
        if xgb is not None:
            base_models["XGBoost"] = xgb

        if len(base_models) >= 2:
            ensemble = EnsembleClassifier()
            ensemble.fit(X_train, y_train, base_models=base_models)
            ensemble.save()
            ens_preds = ensemble.predict(X_test)
            ens_proba = ensemble.predict_proba(X_test)
            all_metrics["Ensemble"] = full_evaluation(y_test, ens_preds, "Ensemble", y_proba=ens_proba)
            del ensemble, ens_preds, ens_proba
        else:
            print("  Not enough base models for ensemble")
    except Exception as e:
        print(f"  Ensemble FAILED: {e}")
    gc.collect()

    # ── 5. Plots ──
    print(f"\n[5/6] Generating plots...")
    try:
        plot_model_comparison(all_metrics)
    except Exception as e:
        print(f"  Model comparison plot failed: {e}")

    # Skip t-SNE (too memory heavy)
    print("  Skipping t-SNE (memory safety)")
    gc.collect()

    try:
        if rf is not None:
            from src.feature_extractor import FeatureExtractor
            fe = FeatureExtractor()
            fnames = fe._build_feature_names()
            rf_imp = rf.feature_importances(fnames)
            plot_feature_importance(rf_imp, model_name="RF", top_n=20)
    except Exception as e:
        print(f"  RF importance plot failed: {e}")

    try:
        if xgb is not None:
            from src.feature_extractor import FeatureExtractor
            fe = FeatureExtractor()
            fnames = fe._build_feature_names()
            xgb_imp = xgb.feature_importances(fnames)
            plot_feature_importance(xgb_imp, model_name="XGBoost", top_n=20)
    except Exception as e:
        print(f"  XGBoost importance plot failed: {e}")

    # ── 6. Summary ──
    print("\n" + "=" * 70)
    print(f"  {'Model':<12s} {'Acc':>7s} {'MacF1':>7s} {'WgtF1':>7s} "
          f"{'MCC':>7s} {'Kappa':>7s} {'AUC':>7s}")
    print("  " + "-" * 58)
    for name, m in all_metrics.items():
        auc = f"{m['auc_roc']:.4f}" if m.get('auc_roc') is not None else "  N/A"
        print(f"  {name:<12s} {m['accuracy']:>7.4f} {m['macro_f1']:>7.4f} "
              f"{m['weighted_f1']:>7.4f} {m['mcc']:>7.4f} {m['kappa']:>7.4f} "
              f"{auc:>7s}")

    # Save summary JSON
    summary_path = RESULTS_DIR / "final_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\n  Summary saved → {summary_path}")
    print("\nDone! Check results/ and results/plots/")


if __name__ == "__main__":
    main()
