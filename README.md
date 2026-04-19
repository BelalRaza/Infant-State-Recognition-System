# Infant Cry Classifier

Automatic classification of infant cry audio into five distress categories —
**hunger**, **belly pain**, **burping**, **discomfort**, and **tiredness** —
using signal-processing features and machine-learning models. The project is
structured in three phases: a classical ML baseline (Phase 1), a deep-learning
model (Phase 2), and a hybrid ensemble that fuses both approaches (Phase 3).

---

## Repository Structure

```
infant-cry-classifier/
├── data/
│   ├── raw/              # original audio files, organised by class
│   │   ├── hunger/
│   │   ├── belly_pain/
│   │   ├── burping/
│   │   ├── discomfort/
│   │   └── tiredness/
│   └── processed/        # cleaned / resampled audio
├── features/             # saved NumPy feature arrays (.npy / .npz)
├── notebooks/            # EDA and feature-engineering notebooks
├── src/                  # Python source modules
│   ├── __init__.py
│   └── config.py         # central project configuration
├── models/               # serialised model files (.pkl / .joblib)
├── results/
│   ├── plots/            # figures (confusion matrices, distributions, …)
│   └── metrics/          # classification reports, scores
├── report/               # LaTeX report source files
├── requirements.txt
├── setup.sh              # creates the directory skeleton
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/infant-cry-classifier.git
cd infant-cry-classifier

# 2. Create and activate a virtual environment (Python 3.10)
python3.10 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` intentionally excludes PyTorch and other
> deep-learning frameworks — those are introduced in Phase 2.

---

## Phase 1 — Classical ML Baseline

### Objective

Build and evaluate two primary classifiers on hand-crafted audio features
extracted from infant cry recordings.

### Feature pipeline

| Step | Detail |
|------|--------|
| Resampling | All clips loaded at native 8 000 Hz (no upsampling) |
| Duration normalisation | Pad or truncate every clip to 7 s (56 000 samples) |
| MFCC extraction | 40 coefficients, FFT window = 512, hop = 160 (~20 ms / ~64 ms at 8 kHz) |
| Statistical aggregation | Mean, std, min, max, skew, kurtosis per coefficient |
| Supplementary features | Zero-crossing rate, spectral centroid, spectral bandwidth, RMS energy |

### Models

| Model | Role | Library |
|-------|------|---------|
| **Gaussian Mixture Model (GMM)** | Density-based generative classifier — one GMM per class | `scikit-learn` |
| **Support Vector Machine (SVM)** | Discriminative classifier with SMOTE + OvO + grid search | `scikit-learn` |
| **Hidden Markov Model (HMM)** | Left-right topology (8 states), captures temporal dynamics | `hmmlearn` |
| **Random Forest (RF)** | Ensemble of 500 decision trees with balanced subsampling | `scikit-learn` |
| **XGBoost** | Gradient boosting with class-weighted loss | `xgboost` |
| **Stacking Ensemble** | Meta-learner combining SVM + RF + XGBoost probabilities | `scikit-learn` |

### Evaluation metrics

- Per-class Precision, Recall, F1-score
- Macro-averaged and weighted F1
- Confusion matrix (normalised)

### Signal analysis checks

- Stationarity testing (Augmented Dickey–Fuller)
- Gaussianity testing (Shapiro–Wilk, D'Agostino–Pearson)
- Feature correlation heatmaps

---

## Phase 2 — Deep Learning *(coming soon)*

Phase 2 will introduce a convolutional or recurrent neural network trained on
Mel-spectrogram representations of the audio. Model interpretability tools
(Grad-CAM, SHAP) will be applied to understand what spectral patterns drive
predictions.

---

## Phase 3 — Hybrid System *(coming soon)*

Phase 3 will fuse the classical ML baseline with the deep-learning model into a
hybrid ensemble. An ablation study will quantify the contribution of each
component, and the full system will be benchmarked against the individual
models.

---

## Results (Phase 1 — Leak-Free Evaluation)

Evaluated on 92 original recordings (no augmented data in test set).
Split-before-augment protocol ensures test-set integrity.

| Model | Accuracy | Macro F1 | Weighted F1 | MCC | AUC-ROC |
|-------|----------|----------|-------------|-----|---------|
| **SVM** | 0.815 | **0.270** | 0.783 | 0.216 | 0.707 |
| Ensemble | **0.837** | 0.249 | 0.780 | 0.155 | 0.599 |
| GMM | 0.641 | 0.207 | 0.670 | −0.045 | 0.454 |
| RF | 0.804 | 0.179 | 0.751 | 0.013 | 0.632 |
| XGBoost | 0.804 | 0.178 | 0.746 | −0.045 | 0.661 |
| HMM | 0.674 | 0.162 | 0.678 | −0.080 | — |

**Note:** Initial results (XGBoost: 96.5% accuracy) were inflated by
data leakage — augmented variants of the same recording appeared in
both train and test sets. The corrected results above reflect true
generalisation performance under extreme class imbalance (382 hunger
vs 8 burping originals).

---

## References

1. Dunstan, P. (2006). *Dunstan Baby Language* — five universal cry categories.
2. Ji, C. et al. (2021). A review of infant cry analysis and classification. *EURASIP J. Audio Speech Music Process.*
3. Librosa documentation: https://librosa.org/doc/latest/

---

## License

This project is developed for academic purposes as part of a university
coursework submission.
