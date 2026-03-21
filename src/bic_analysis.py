"""
bic_analysis.py — BIC analysis for GMM component selection.

Computes the Bayesian Information Criterion (BIC) for a range of GMM
component counts to justify the choice of n_components=4 used in the
GMM classifier.

Usage
-----
    python -m src.bic_analysis          # from project root
    python src/bic_analysis.py          # same effect
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSES, FEATURES_DIR, PLOTS_DIR, RAW_DIR, RANDOM_STATE,
)
from src.data_loader import load_dataset, get_class_distribution
from src.feature_extractor import FeatureExtractor
from src.augmentation import run_augmentation


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
    if not (X_path.exists() and y_path.exists()):
        return False
    try:
        cached_n = np.load(X_path, mmap_mode="r").shape[0]
        return cached_n == expected_n
    except Exception:
        return False


def run_bic_analysis(
    component_range: range = range(1, 11),
    covariance_type: str = "diag",
) -> None:
    """Compute per-class BIC for each n_components and plot the results."""

    print("=" * 60)
    print("  BIC Analysis — GMM Component Selection")
    print("=" * 60)

    # ── Load features ────────────────────────────────────────────
    print("\n[1/3] Loading features …")
    run_augmentation()

    X_path = FEATURES_DIR / "X.npy"
    y_path = FEATURES_DIR / "y.npy"
    n_wav = _count_wav_files()

    if _cache_is_valid(X_path, y_path, n_wav):
        print(f"  Loading cached features ({n_wav} samples) …")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        print(f"  Cache stale — re-extracting features …")
        dataset_list, _ = load_dataset(return_list=True)
        extractor = FeatureExtractor()
        X, y, _ = extractor.extract_and_save_dataset(dataset_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Compute BIC for each class and component count ───────────
    print("\n[2/3] Computing BIC scores …")
    bic_results = {}  # class_name -> list of BIC values

    for idx, cls in enumerate(CLASSES):
        mask = y == idx
        X_cls = X_scaled[mask]
        n_samples = X_cls.shape[0]

        bics = []
        for n_comp in component_range:
            if n_comp > n_samples:
                bics.append(np.nan)
                continue

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=covariance_type,
                max_iter=200,
                reg_covar=1e-3,
                random_state=RANDOM_STATE,
                n_init=3,
            )
            gmm.fit(X_cls)
            bics.append(gmm.bic(X_cls))

        bic_results[cls] = bics
        best_k = list(component_range)[np.nanargmin(bics)]
        print(f"  {cls:12s}  n={n_samples:4d}  best_k={best_k}  "
              f"BIC_min={min(b for b in bics if not np.isnan(b)):.0f}")

    # ── Also compute overall BIC (all data pooled) ───────────────
    overall_bics = []
    for n_comp in component_range:
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=covariance_type,
            max_iter=200,
            reg_covar=1e-3,
            random_state=RANDOM_STATE,
            n_init=3,
        )
        gmm.fit(X_scaled)
        overall_bics.append(gmm.bic(X_scaled))

    best_overall = list(component_range)[np.argmin(overall_bics)]
    print(f"  {'overall':12s}  n={len(y):4d}  best_k={best_overall}  "
          f"BIC_min={min(overall_bics):.0f}")

    # ── Plot ─────────────────────────────────────────────────────
    print("\n[3/3] Plotting BIC curves …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: per-class BIC
    ax1 = axes[0]
    comp_list = list(component_range)
    for cls, bics in bic_results.items():
        ax1.plot(comp_list, bics, marker="o", label=cls, linewidth=1.5)
    ax1.set_xlabel("Number of GMM Components")
    ax1.set_ylabel("BIC (lower is better)")
    ax1.set_title("Per-Class BIC vs. Number of Components")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(comp_list)

    # Right panel: overall BIC
    ax2 = axes[1]
    ax2.plot(comp_list, overall_bics, marker="s", color="black",
             linewidth=2, label="All classes pooled")
    ax2.axvline(x=best_overall, color="red", linestyle="--", alpha=0.7,
                label=f"Best k={best_overall}")
    ax2.axvline(x=4, color="blue", linestyle=":", alpha=0.7,
                label="Chosen k=4")
    ax2.set_xlabel("Number of GMM Components")
    ax2.set_ylabel("BIC (lower is better)")
    ax2.set_title("Overall BIC vs. Number of Components")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(comp_list)

    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "bic_component_selection.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  BIC plot saved → {save_path}")

    print("\nDone.")


if __name__ == "__main__":
    run_bic_analysis()
