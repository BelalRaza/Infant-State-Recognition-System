"""
feature_extractor.py — Feature extraction for infant cry classification.

This module provides the ``FeatureExtractor`` class, which produces a
**246-dimensional** feature vector per audio clip:

    40 MFCCs  x 2 stats (mean, std)        = 80
    40 Deltas x 2 stats                     = 80
    40 Delta-deltas x 2 stats               = 80
    6 spectral / energy descriptors         =  6
    ─────────────────────────────────────────
    Total                                    246

The 240-dim MFCC block captures the static and dynamic spectral envelope
of the infant's vocal tract.  The 6 statistical features add physically
interpretable descriptors of noisiness, brightness, loudness, tonality,
and spectral shape.

Relationship to ``feature_extraction.py``
-----------------------------------------
The older ``feature_extraction.py`` module uses 6 summary statistics per
coefficient (mean, std, min, max, skew, kurtosis) producing a 750-dim
vector.  This module uses only mean + std (2 stats), which is more compact,
less prone to overfitting on small classes, and aligns with the standard
MFCC feature convention in speech/audio literature.  Both modules coexist:
``feature_extraction.py`` is used by the existing ``run_pipeline.py``, while
this module is the primary feature extractor for the Phase 1 notebooks and
the rubric deliverable.
"""

import pickle
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import (
    SAMPLE_RATE,
    N_MFCC,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    FEATURES_DIR,
)


class FeatureExtractor:
    """Extract MFCC and statistical audio features for classical ML.

    This class is the single entry point for converting raw audio waveforms
    into fixed-length feature vectors suitable for GMM, SVM, and HMM
    classifiers.

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

    Examples
    --------
    >>> extractor = FeatureExtractor()
    >>> features, names = extractor.extract_all(audio_array, sr=8000)
    >>> features.shape
    (246,)
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

        MFCCs capture the spectral envelope of sound, which encodes vocal
        tract shape.  Delta and delta-delta capture how the spectrum changes
        over time — important for cry dynamics.  We use mean+std to summarise
        temporal variation into a fixed-length feature vector.

        Pipeline
        --------
        1. Compute 40 MFCCs from the Mel-scaled power spectrogram.
           Each coefficient corresponds to a different frequency band on
           the perceptual Mel scale — lower coefficients capture the broad
           spectral shape, higher ones capture finer detail.
        2. Compute first-order temporal derivative (delta) of MFCCs.
           These encode the *velocity* of spectral change between adjacent
           frames, capturing transitions like the onset/offset of a cry.
        3. Compute second-order temporal derivative (delta-delta) of MFCCs.
           These encode the *acceleration* of spectral change, capturing
           how abruptly the cry dynamics shift.
        4. For each coefficient in each of the three matrices, compute the
           mean and standard deviation over all time frames.  Mean captures
           the average spectral profile; std captures how much it varies
           across the clip.

        Parameters
        ----------
        audio : np.ndarray
            1-D waveform array (float32).
        sr : int, optional
            Sample rate.  Defaults to ``self.sr``.

        Returns
        -------
        np.ndarray, shape (240,)
            Concatenation of [mfcc_mean, mfcc_std, delta_mean, delta_std,
            delta2_mean, delta2_std], each of length n_mfcc (40).
            Total: 40 * 3 * 2 = 240.
        """
        if sr is None:
            sr = self.sr

        # Step 1 — static MFCCs: shape (n_mfcc, T)
        # Each row is one cepstral coefficient over T time frames.
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )

        # Step 2 — first-order derivative (velocity): how fast each
        # coefficient changes between adjacent frames.
        delta = librosa.feature.delta(mfcc, order=1)

        # Step 3 — second-order derivative (acceleration): rate of change
        # of the delta — captures onset/offset dynamics of the cry.
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Step 4 — summarise each coefficient's time series with mean + std.
        # Mean captures the average spectral profile of the cry;
        # std captures how much the spectrum varies within the clip.
        mfcc_mean = np.mean(mfcc, axis=1)       # (40,)
        mfcc_std = np.std(mfcc, axis=1)          # (40,)
        delta_mean = np.mean(delta, axis=1)      # (40,)
        delta_std = np.std(delta, axis=1)        # (40,)
        delta2_mean = np.mean(delta2, axis=1)    # (40,)
        delta2_std = np.std(delta2, axis=1)      # (40,)

        return np.concatenate([
            mfcc_mean, mfcc_std,
            delta_mean, delta_std,
            delta2_mean, delta2_std,
        ]).astype(np.float32)

    # ── Statistical / spectral features ───────────────────────────────

    def extract_statistical_features(
        self, audio: np.ndarray, sr: int = None,
    ) -> dict:
        """Extract physically interpretable spectral and energy descriptors.

        Each feature captures a distinct acoustic property of the cry signal:

        - **zero_crossing_rate**: Mean rate at which the waveform crosses the
          zero axis — higher values indicate noisier, higher-frequency content
          (e.g. breathy burping sounds vs tonal hunger cries).
        - **spectral_centroid**: Centre of mass of the power spectrum,
          perceived as the "brightness" of the sound — belly-pain cries tend
          to have a higher centroid than tired whimpers.
        - **spectral_rolloff**: Frequency below which 85 % of the total
          spectral energy is concentrated — characterises the upper frequency
          limit of the dominant harmonic content.
        - **spectral_bandwidth**: Weighted standard deviation of frequencies
          around the centroid — measures how spread out the spectrum is;
          wider bandwidth often indicates a more distressed, strained cry.
        - **rms_energy**: Root-mean-square amplitude averaged over analysis
          frames — a direct measure of perceived loudness, useful for
          distinguishing loud pain cries from quiet tiredness whimpers.
        - **spectral_flatness**: Ratio of the geometric mean to the
          arithmetic mean of the power spectrum — values near 1.0 indicate
          noise-like signals, values near 0.0 indicate tonal / harmonic
          content.  Harmonic cries (hunger) score low; noisy grunts
          (burping) score higher.

        Parameters
        ----------
        audio : np.ndarray
            1-D waveform array (float32).
        sr : int, optional
            Sample rate.  Defaults to ``self.sr``.

        Returns
        -------
        dict
            Keys are feature names, values are scalar floats.
            Order: zero_crossing_rate, spectral_centroid, spectral_rolloff,
            spectral_bandwidth, rms_energy, spectral_flatness.
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
        """Concatenate MFCC vector + statistical features into one flat array.

        This is the primary interface for downstream models.  It produces
        a single 246-dimensional vector that encodes the full acoustic
        profile of one audio clip.

        Parameters
        ----------
        audio : np.ndarray
            1-D waveform array.
        sr : int, optional
            Sample rate.

        Returns
        -------
        features : np.ndarray, shape (246,)
            Flat feature vector: 240 MFCC-based + 6 spectral/energy.
        feature_names : list[str]
            Human-readable name for each dimension (length 246).
        """
        if sr is None:
            sr = self.sr

        mfcc_vec = self.extract_mfcc(audio, sr)                   # (240,)
        stat_dict = self.extract_statistical_features(audio, sr)  # 6 scalars
        stat_vec = np.array(list(stat_dict.values()), dtype=np.float32)

        features = np.concatenate([mfcc_vec, stat_vec])           # (246,)
        feature_names = self._build_feature_names(stat_dict)

        return features, feature_names

    def _build_feature_names(self, stat_dict: dict = None) -> list[str]:
        """Generate a list of human-readable names for every feature dimension.

        The order matches the concatenation in ``extract_all()``:
        [mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std,
         zcr, centroid, rolloff, bandwidth, rms, flatness].

        Returns
        -------
        list[str], length 246
        """
        names = []

        # MFCC block: 3 types x 2 stats x 40 coefficients = 240
        for prefix in ["mfcc", "delta", "delta2"]:
            for stat in ["mean", "std"]:
                for i in range(self.n_mfcc):
                    names.append(f"{prefix}_{i}_{stat}")

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
        """Extract features for every sample in the dataset and save to disk.

        Iterates over the list-of-dicts produced by
        ``InfantCryLoader.load_dataset()``, extracts a 246-dim feature
        vector per clip, and persists three files:

        - ``X.npy``  — feature matrix, shape (n_samples, 246)
        - ``y.npy``  — integer label array, shape (n_samples,)
        - ``feature_names.pkl`` — ordered list of 246 feature names

        A tqdm progress bar is displayed during extraction.  Corrupted
        samples that raise exceptions are logged and skipped so that one
        bad file does not abort the entire batch.

        Parameters
        ----------
        dataset_list : list[dict]
            Each element must have keys ``"audio"`` (np.ndarray),
            ``"label_idx"`` (int), and ``"filepath"`` (str).
            This is the output format of ``InfantCryLoader.load_dataset()``.
        save_path : str or Path, optional
            Directory to write output files into.
            Defaults to ``config.FEATURES_DIR``.

        Returns
        -------
        X : np.ndarray, shape (n_samples, 246)
        y : np.ndarray, shape (n_samples,)
        feature_names : list[str], length 246
        """
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

        # Persist to disk for reproducible downstream usage
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
