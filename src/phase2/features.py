"""Feature extraction: mel-spectrograms and 32-dim domain features."""

import librosa
import numpy as np

from src.phase2.config import (
    SAMPLE_RATE, MEL_N_FFT, MEL_HOP, MEL_N_MELS,
    MEL_FMIN, MEL_FMAX, N_MFCC_DOMAIN,
)


def extract_mel_spectrogram(waveform, sr=SAMPLE_RATE):
    """Extract log-mel spectrogram. Returns shape (n_mels, time_frames)."""
    S = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        n_fft=MEL_N_FFT, hop_length=MEL_HOP,
        n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX,
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def extract_domain_features(waveform, sr=SAMPLE_RATE):
    """Extract 32-dim handcrafted feature vector.

    Components: F0 stats (4) + hyperphonation (1) + HNR (1) +
    jitter (1) + shimmer (1) + MFCC 1-13 means (13) + delta MFCC 1-12 means (12) = 32
    """
    feats = []

    # --- F0 contour via pyin ---
    f0, voiced, _ = librosa.pyin(
        waveform, fmin=80, fmax=800, sr=sr, hop_length=MEL_HOP,
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if len(f0_valid) > 1:
        feats.append(np.mean(f0_valid))
        feats.append(np.std(f0_valid))
        feats.append(np.max(f0_valid) - np.min(f0_valid))
        feats.append(np.mean(f0_valid >= 1000.0))  # hyperphonation %
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    # --- HNR (harmonics-to-noise ratio via autocorrelation) ---
    feats.append(_compute_hnr(waveform, sr))

    # --- Jitter (F0 perturbation) ---
    feats.append(_compute_jitter(f0_valid))

    # --- Shimmer (amplitude perturbation) ---
    feats.append(_compute_shimmer(waveform, sr, f0_valid))

    # --- MFCC 1-13 means ---
    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=N_MFCC_DOMAIN,
        n_fft=MEL_N_FFT, hop_length=MEL_HOP, n_mels=MEL_N_MELS,
    )
    feats.extend(np.mean(mfcc[1:14], axis=1).tolist())

    # --- Delta MFCC 1-12 means ---
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    feats.extend(np.mean(delta_mfcc[1:13], axis=1).tolist())

    return np.array(feats, dtype=np.float32)


def _compute_hnr(waveform, sr):
    """Estimate HNR from autocorrelation."""
    frame_len = int(0.04 * sr)
    hop = int(0.01 * sr)
    hnr_vals = []
    for start in range(0, len(waveform) - frame_len, hop):
        frame = waveform[start:start + frame_len]
        if np.max(np.abs(frame)) < 1e-6:
            continue
        r = np.correlate(frame, frame, mode="full")
        r = r[len(r) // 2:]
        if len(r) < 2:
            continue
        r0 = r[0]
        if r0 < 1e-10:
            continue
        peak = np.max(r[1:])
        ratio = peak / r0
        ratio = np.clip(ratio, 1e-10, 1.0 - 1e-10)
        hnr_vals.append(10 * np.log10(ratio / (1 - ratio)))
    return float(np.mean(hnr_vals)) if hnr_vals else 0.0


def _compute_jitter(f0_valid):
    """Local jitter: mean absolute difference of consecutive F0 periods."""
    if len(f0_valid) < 2:
        return 0.0
    periods = 1.0 / np.clip(f0_valid, 1.0, None)
    diffs = np.abs(np.diff(periods))
    return float(np.mean(diffs) / np.mean(periods)) if np.mean(periods) > 0 else 0.0


def _compute_shimmer(waveform, sr, f0_valid):
    """Local shimmer: mean absolute difference of consecutive amplitudes."""
    if len(f0_valid) < 2:
        return 0.0
    frame_len = int(0.03 * sr)
    hop = int(0.01 * sr)
    amps = []
    for start in range(0, len(waveform) - frame_len, hop):
        amps.append(np.max(np.abs(waveform[start:start + frame_len])))
    if len(amps) < 2:
        return 0.0
    amps = np.array(amps)
    amps = amps[amps > 1e-6]
    if len(amps) < 2:
        return 0.0
    diffs = np.abs(np.diff(amps))
    return float(np.mean(diffs) / np.mean(amps))


def _compute_spectral_centroid_mean(waveform, sr):
    """Mean spectral centroid."""
    sc = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=MEL_HOP)
    return float(np.mean(sc))
