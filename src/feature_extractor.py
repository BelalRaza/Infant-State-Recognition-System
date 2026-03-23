"""
feature_extractor.py — Feature extraction for infant cry classification.

This module provides the ``FeatureExtractor`` class, which produces a
**411-dimensional** feature vector per audio clip:

    40 MFCCs  x 2 stats (mean, std)        = 80  \
    40 Deltas x 2 stats                     = 80   } 240 MFCC block
    40 Delta-deltas x 2 stats               = 80  /
    20 CQCCs x 2 stats (mean, std)          = 40  \
    20 Delta-CQCCs x 2 stats                = 40   } 120 CQCC block
    20 Delta2-CQCCs x 2 stats               = 40  /
    7 F0 / pitch contour features           =  7
    7 spectral contrast bands x 2 stats     = 14
    12 chroma bins x 2 stats                = 24
    6 spectral / energy descriptors         =  6
    ─────────────────────────────────────────
    Total                                    411

The CQCC block uses Constant-Q Transform which provides variable
time-frequency resolution better suited to infant cry harmonics than
the fixed Mel scale used by MFCCs.
"""

import pickle
import librosa
import numpy as np
from pathlib import Path
from scipy.fft import dct
from tqdm import tqdm

from src.config import (
    SAMPLE_RATE,
    N_MFCC,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    CQT_N_BINS,
    CQT_BINS_PER_OCTAVE,
    N_CQCC,
    PITCH_FMIN,
    PITCH_FMAX,
    N_CONTRAST_BANDS,
    FEATURES_DIR,
)


class FeatureExtractor:
    """Extract MFCC, CQCC, pitch, contrast, chroma, and spectral features.

    Parameters
    ----------
    n_mfcc : int
        Number of Mel-frequency cepstral coefficients (default 40).
    sr : int
        Expected sample rate of input audio (default from config).
    hop_length : int
        STFT hop length in samples (default from config).
    n_fft : int
        FFT window size in samples (default from config).
    n_mels : int
        Number of Mel filter banks (default from config).
    """

    def __init__(
        self,
        n_mfcc: int = N_MFCC,
        sr: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        n_mels: int = N_MELS,
    ):
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels

    # ── MFCC features ─────────────────────────────────────────────────

    def extract_mfcc(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract MFCC, delta-MFCC, and delta-delta-MFCC summary features.

        Returns
        -------
        np.ndarray, shape (240,)
            Concatenation of [mfcc_mean, mfcc_std, delta_mean, delta_std,
            delta2_mean, delta2_std], each of length n_mfcc (40).
        """
        if sr is None:
            sr = self.sr

        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )

        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        delta_mean = np.mean(delta, axis=1)
        delta_std = np.std(delta, axis=1)
        delta2_mean = np.mean(delta2, axis=1)
        delta2_std = np.std(delta2, axis=1)

        return np.concatenate([
            mfcc_mean, mfcc_std,
            delta_mean, delta_std,
            delta2_mean, delta2_std,
        ]).astype(np.float32)

    # ── CQCC features (Constant-Q Cepstral Coefficients) ──────────────

    def extract_cqcc(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract CQCC features using Constant-Q Transform.

        CQT provides variable time-frequency resolution — higher frequency
        resolution at low frequencies and higher time resolution at high
        frequencies.  This better captures infant cry harmonics than the
        fixed Mel scale used by MFCCs.

        Returns
        -------
        np.ndarray, shape (120,)
            [cqcc_mean, cqcc_std, delta_mean, delta_std,
             delta2_mean, delta2_std], each of length N_CQCC (20).
        """
        if sr is None:
            sr = self.sr

        # Compute CQT magnitude spectrogram
        # Limit bins to stay under Nyquist (sr/2). At 8kHz, max ~4kHz.
        safe_bins = min(CQT_N_BINS, int(CQT_BINS_PER_OCTAVE * np.log2(sr / 2 / 32.7)))
        C = np.abs(librosa.cqt(
            y=audio, sr=sr, hop_length=self.hop_length,
            n_bins=safe_bins, bins_per_octave=CQT_BINS_PER_OCTAVE,
        ))

        # Convert to log power
        C_db = librosa.amplitude_to_db(C, ref=np.max)

        # Apply DCT to get cepstral coefficients (analogous to MFCC from Mel)
        cqcc = dct(C_db, type=2, axis=0, norm='ortho')[:N_CQCC]  # (20, T)

        # Compute deltas
        delta_cqcc = librosa.feature.delta(cqcc, order=1)
        delta2_cqcc = librosa.feature.delta(cqcc, order=2)

        return np.concatenate([
            np.mean(cqcc, axis=1), np.std(cqcc, axis=1),             # 40
            np.mean(delta_cqcc, axis=1), np.std(delta_cqcc, axis=1), # 40
            np.mean(delta2_cqcc, axis=1), np.std(delta2_cqcc, axis=1), # 40
        ]).astype(np.float32)  # Total: 120

    # ── F0 / pitch contour features ──────────────────────────────────

    def extract_pitch_features(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract fundamental frequency (F0) contour features.

        F0 is the dominant pitch of the cry.  Different cry types exhibit
        different pitch patterns — pain cries tend to be higher pitched,
        tiredness cries lower and more monotone.

        Returns
        -------
        np.ndarray, shape (7,)
            [f0_mean, f0_std, f0_max, f0_min, f0_range, voiced_fraction, f0_slope]
        """
        if sr is None:
            sr = self.sr

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=PITCH_FMIN, fmax=PITCH_FMAX,
            sr=sr, hop_length=self.hop_length,
        )

        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        if len(f0_valid) > 1:
            f0_mean = float(np.mean(f0_valid))
            f0_std = float(np.std(f0_valid))
            f0_max = float(np.max(f0_valid))
            f0_min = float(np.min(f0_valid))
            f0_range = f0_max - f0_min
            f0_slope = float(np.polyfit(range(len(f0_valid)), f0_valid, 1)[0])
        elif len(f0_valid) == 1:
            f0_mean = float(f0_valid[0])
            f0_std = 0.0
            f0_max = f0_mean
            f0_min = f0_mean
            f0_range = 0.0
            f0_slope = 0.0
        else:
            f0_mean = f0_std = f0_max = f0_min = f0_range = f0_slope = 0.0

        voiced_fraction = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0

        return np.array([
            f0_mean, f0_std, f0_max, f0_min, f0_range,
            voiced_fraction, f0_slope,
        ], dtype=np.float32)

    # ── Spectral contrast features ───────────────────────────────────

    def extract_spectral_contrast(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract spectral contrast (peak-valley difference per sub-band).

        Returns
        -------
        np.ndarray, shape (14,)
            Mean and std of 7 contrast bands (6 bands + 1 valley).
        """
        if sr is None:
            sr = self.sr

        safe_bands = min(N_CONTRAST_BANDS, 4)  # safe for 8kHz
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr,
            n_bands=safe_bands,
            fmin=100,
            hop_length=self.hop_length,
        )  # shape (safe_bands+1, T)

        stats = np.concatenate([
            np.mean(contrast, axis=1),
            np.std(contrast, axis=1),
        ]).astype(np.float32)
        # Pad to 14 dims for consistency
        result = np.zeros(14, dtype=np.float32)
        result[:len(stats)] = stats
        return result

    # ── Chroma features ──────────────────────────────────────────────

    def extract_chroma(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract chroma (pitch class) features.

        Returns
        -------
        np.ndarray, shape (24,)
            Mean and std of 12 chroma bins.
        """
        if sr is None:
            sr = self.sr

        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, hop_length=self.hop_length,
        )  # shape (12, T)

        return np.concatenate([
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
        ]).astype(np.float32)  # 24

    # ── Statistical / spectral features ───────────────────────────────

    def extract_statistical_features(
        self, audio: np.ndarray, sr: int = None,
    ) -> dict:
        """Extract physically interpretable spectral and energy descriptors.

        Returns
        -------
        dict
            Keys are feature names, values are scalar floats (6 features).
        """
        if sr is None:
            sr = self.sr

        zcr = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length,
        )
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=self.hop_length,
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=self.hop_length, roll_percent=0.85,
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, hop_length=self.hop_length,
        )
        rms = librosa.feature.rms(
            y=audio, hop_length=self.hop_length,
        )
        flatness = librosa.feature.spectral_flatness(
            y=audio, hop_length=self.hop_length,
        )

        return {
            "zero_crossing_rate": float(np.mean(zcr)),
            "spectral_centroid": float(np.mean(centroid)),
            "spectral_rolloff": float(np.mean(rolloff)),
            "spectral_bandwidth": float(np.mean(bandwidth)),
            "rms_energy": float(np.mean(rms)),
            "spectral_flatness": float(np.mean(flatness)),
        }

    # ── Combined feature vector ───────────────────────────────────────

    def extract_all(
        self, audio: np.ndarray, sr: int = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Concatenate all feature blocks into one flat array.

        Returns
        -------
        features : np.ndarray, shape (411,)
        feature_names : list[str]
        """
        if sr is None:
            sr = self.sr

        mfcc_vec = self.extract_mfcc(audio, sr)                       # (240,)
        cqcc_vec = self.extract_cqcc(audio, sr)                       # (120,)
        pitch_vec = self.extract_pitch_features(audio, sr)            # (7,)
        contrast_vec = self.extract_spectral_contrast(audio, sr)      # (14,)
        chroma_vec = self.extract_chroma(audio, sr)                   # (24,)
        stat_dict = self.extract_statistical_features(audio, sr)      # 6 scalars
        stat_vec = np.array(list(stat_dict.values()), dtype=np.float32)

        features = np.concatenate([
            mfcc_vec, cqcc_vec, pitch_vec,
            contrast_vec, chroma_vec, stat_vec,
        ])  # (411,)

        feature_names = self._build_feature_names(stat_dict)
        return features, feature_names

    def _build_feature_names(self, stat_dict: dict = None) -> list[str]:
        """Generate human-readable names for every feature dimension."""
        names = []

        # MFCC block: 3 types x 2 stats x 40 coefficients = 240
        for prefix in ["mfcc", "delta", "delta2"]:
            for stat in ["mean", "std"]:
                for i in range(self.n_mfcc):
                    names.append(f"{prefix}_{i}_{stat}")

        # CQCC block: 3 types x 2 stats x 20 coefficients = 120
        for prefix in ["cqcc", "delta_cqcc", "delta2_cqcc"]:
            for stat in ["mean", "std"]:
                for i in range(N_CQCC):
                    names.append(f"{prefix}_{i}_{stat}")

        # Pitch features: 7
        names.extend([
            "f0_mean", "f0_std", "f0_max", "f0_min", "f0_range",
            "voiced_fraction", "f0_slope",
        ])

        # Spectral contrast: 14 (7 bands x 2 stats)
        for stat in ["mean", "std"]:
            for i in range(N_CONTRAST_BANDS + 1):
                names.append(f"spectral_contrast_{i}_{stat}")

        # Chroma: 24 (12 bins x 2 stats)
        for stat in ["mean", "std"]:
            for i in range(12):
                names.append(f"chroma_{i}_{stat}")

        # Statistical / spectral features: 6
        if stat_dict is not None:
            names.extend(stat_dict.keys())
        else:
            names.extend([
                "zero_crossing_rate", "spectral_centroid",
                "spectral_rolloff", "spectral_bandwidth",
                "rms_energy", "spectral_flatness",
            ])

        return names

    # ── Batch extraction + persistence ────────────────────────────────

    def extract_and_save_dataset(
        self,
        dataset_list: list[dict],
        save_path: str | Path = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract features for every sample in the dataset and save to disk."""
        if save_path is None:
            save_path = FEATURES_DIR
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        X_list = []
        y_list = []
        feature_names = None
        corrupted = []

        for sample in tqdm(dataset_list, desc="Extracting features"):
            try:
                feats, names = self.extract_all(sample["audio"], self.sr)
                X_list.append(feats)
                y_list.append(sample["label_idx"])
                if feature_names is None:
                    feature_names = names
            except Exception as exc:
                fpath = sample.get("filepath", "unknown")
                corrupted.append(fpath)
                print(f"[ERROR] Feature extraction failed for {fpath}: {exc}")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        np.save(save_path / "X.npy", X)
        np.save(save_path / "y.npy", y)
        with open(save_path / "feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)

        print(f"\nFeatures saved to {save_path}/")
        print(f"  X.npy            : shape {X.shape}")
        print(f"  y.npy            : shape {y.shape}")
        print(f"  feature_names.pkl: {len(feature_names)} names")

        if corrupted:
            print(f"\n[WARN] {len(corrupted)} sample(s) failed:")
            for p in corrupted:
                print(f"  - {p}")

        return X, y, feature_names
