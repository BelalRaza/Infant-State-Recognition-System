"""
statistical_tests.py — Stationarity and Gaussianity checks for audio features.

These tests are required in Phase 1 to demonstrate understanding of the
assumptions underlying GMM (Gaussianity) and time-series modelling (stationarity).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from pathlib import Path

from src.config import CLASSES, PLOTS_DIR, METRICS_DIR


# ── Stationarity ──────────────────────────────────────────────────────


def augmented_dickey_fuller(series: np.ndarray) -> dict:
    """Run the Augmented Dickey–Fuller test on a 1-D signal.

    Null hypothesis: the series has a unit root (non-stationary).
    A p-value < 0.05 suggests the series IS stationary.

    Returns
    -------
    dict with keys: adf_statistic, p_value, n_lags, n_obs,
                    critical_values, is_stationary
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series, autolag="AIC")
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "n_lags": int(result[2]),
        "n_obs": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "is_stationary": result[1] < 0.05,
    }


def batch_stationarity_test(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_samples_per_class: int = 10,
    classes: list = None,
) -> pd.DataFrame:
    """Run ADF test on a subset of waveforms and return a summary DataFrame.

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_clips, n_audio_samples)
    y : np.ndarray, shape (n_clips,)
    n_samples_per_class : int
        How many clips per class to test (for speed).
    classes : list[str]

    Returns
    -------
    pd.DataFrame with columns: class, sample_idx, adf_statistic, p_value,
                                is_stationary
    """
    if classes is None:
        classes = CLASSES

    rows = []
    for idx, cls in enumerate(classes):
        mask = np.where(y == idx)[0]
        chosen = mask[:n_samples_per_class]
        for i, sample_idx in enumerate(chosen):
            try:
                result = augmented_dickey_fuller(X_raw[sample_idx])
                rows.append({
                    "class": cls,
                    "sample_idx": int(sample_idx),
                    "adf_statistic": result["adf_statistic"],
                    "p_value": result["p_value"],
                    "is_stationary": result["is_stationary"],
                })
            except Exception as exc:
                rows.append({
                    "class": cls,
                    "sample_idx": int(sample_idx),
                    "adf_statistic": np.nan,
                    "p_value": np.nan,
                    "is_stationary": None,
                    "error": str(exc),
                })

    df = pd.DataFrame(rows)
    return df


def plot_stationarity_summary(
    df: pd.DataFrame,
    save: bool = True,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Bar chart showing fraction of stationary clips per class."""
    summary = (
        df.groupby("class")["is_stationary"]
        .mean()
        .reindex(CLASSES)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary.index, summary.values, color="teal", edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction Stationary (ADF p < 0.05)")
    ax.set_title("Stationarity Check — Augmented Dickey–Fuller Test")
    for bar, val in zip(bars, summary.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=10)
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "stationarity_adf.png"
        fig.savefig(path, dpi=150)
        print(f"Plot saved → {path}")
    plt.close(fig)


# ── Gaussianity ───────────────────────────────────────────────────────


def test_gaussianity_shapiro(data: np.ndarray) -> dict:
    """Shapiro–Wilk test for normality on a 1-D array.

    Null hypothesis: the data is drawn from a normal distribution.
    p-value < 0.05 → reject normality.

    Note: Shapiro–Wilk is limited to n <= 5000.  For larger arrays the
    function sub-samples automatically.
    """
    if len(data) > 5000:
        rng = np.random.default_rng(42)
        data = rng.choice(data, size=5000, replace=False)

    stat, p = sp_stats.shapiro(data)
    return {"statistic": float(stat), "p_value": float(p), "is_normal": p >= 0.05}


def test_gaussianity_dagostino(data: np.ndarray) -> dict:
    """D'Agostino–Pearson omnibus test for normality.

    Works on larger samples than Shapiro–Wilk and combines skewness +
    kurtosis tests.
    """
    if len(data) < 20:
        return {"statistic": np.nan, "p_value": np.nan, "is_normal": None}

    stat, p = sp_stats.normaltest(data)
    return {"statistic": float(stat), "p_value": float(p), "is_normal": p >= 0.05}


def batch_gaussianity_test(
    features: np.ndarray,
    y: np.ndarray,
    feature_names: list = None,
    n_features_to_test: int = 10,
    classes: list = None,
) -> pd.DataFrame:
    """Test normality of the first *n_features_to_test* feature dimensions
    within each class.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray
    feature_names : list[str] | None
    n_features_to_test : int
    classes : list[str]

    Returns
    -------
    pd.DataFrame
    """
    if classes is None:
        classes = CLASSES
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(features.shape[1])]

    n_test = min(n_features_to_test, features.shape[1])
    rows = []

    for idx, cls in enumerate(classes):
        mask = y == idx
        X_cls = features[mask]
        if X_cls.shape[0] < 8:
            continue

        for feat_idx in range(n_test):
            col = X_cls[:, feat_idx]

            shapiro = test_gaussianity_shapiro(col)
            dagostino = test_gaussianity_dagostino(col)

            rows.append({
                "class": cls,
                "feature": feature_names[feat_idx],
                "feat_idx": feat_idx,
                "shapiro_stat": shapiro["statistic"],
                "shapiro_p": shapiro["p_value"],
                "shapiro_normal": shapiro["is_normal"],
                "dagostino_stat": dagostino["statistic"],
                "dagostino_p": dagostino["p_value"],
                "dagostino_normal": dagostino["is_normal"],
                "skewness": float(sp_stats.skew(col)),
                "kurtosis": float(sp_stats.kurtosis(col)),
            })

    return pd.DataFrame(rows)


def plot_gaussianity_summary(
    df: pd.DataFrame,
    test: str = "shapiro",
    save: bool = True,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Heatmap of normality test p-values (features x classes)."""
    col = f"{test}_p"
    pivot = df.pivot_table(index="feature", columns="class", values=col)
    pivot = pivot.reindex(columns=CLASSES)

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=0.1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Gaussianity — {test.title()} p-values (green ≥ 0.05 = normal)")
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"gaussianity_{test}.png"
        fig.savefig(path, dpi=150)
        print(f"Plot saved → {path}")
    plt.close(fig)


def plot_feature_distributions(
    features: np.ndarray,
    y: np.ndarray,
    feature_indices: list = None,
    classes: list = None,
    save: bool = True,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """KDE plots of selected features, coloured by class."""
    if classes is None:
        classes = CLASSES
    if feature_indices is None:
        feature_indices = list(range(min(6, features.shape[1])))

    n = len(feature_indices)
    cols = 3
    rows_fig = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_fig, cols, figsize=(5 * cols, 4 * rows_fig))
    axes = np.atleast_2d(axes)

    for i, feat_idx in enumerate(feature_indices):
        ax = axes[i // cols, i % cols]
        for cls_idx, cls in enumerate(classes):
            mask = y == cls_idx
            ax.hist(
                features[mask, feat_idx],
                bins=30,
                alpha=0.5,
                density=True,
                label=cls,
            )
        ax.set_title(f"Feature {feat_idx}")
        ax.legend(fontsize=7)

    for j in range(i + 1, rows_fig * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle("Feature Distributions by Class", fontsize=14, y=1.01)
    plt.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "feature_distributions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {path}")
    plt.close(fig)
