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

## Phase 2 — Deep Learning & Hybrid Ensemble

Phase 2 introduces a CNN+BiLSTM+Temporal Attention architecture on 64-band
mel-spectrograms with LDAM loss, Deferred Re-Weighting, domain feature fusion,
and knowledge distillation (135K Teacher → 16K Student). Data split uses
GroupShuffleSplit by infant UUID (70/15/15).

DL alone could not beat the Phase 1 SVM (best DL: Teacher macro-F1 = 0.293).
The breakthrough came from a **hybrid weighted ensemble** fusing SVM + DL
probability outputs.

### Key techniques
- LDAM loss with class-dependent margins
- Deferred Re-Weighting (uniform → class-balanced at epoch 60)
- Mixup, SpecAugment, and balanced augmentation (326 → 1920 samples)
- Knowledge distillation + INT8 quantisation (35.9 KB, 11.4 ms inference)

---

## Phase 3 — Edge Deployment *(planned)*

Phase 3 will deploy the INT8-quantised student model via TFLite Micro on ESP32
and ONNX.js for browser-based inference, with on-device mel-spectrogram
extraction.

---

## Results

### Phase 1 — Classical ML (leak-free, 92-sample test set)

| Model | Accuracy | Macro F1 | MCC | AUC-ROC |
|-------|----------|----------|-----|---------|
| **SVM (SMOTE)** | 0.815 | **0.270** | 0.216 | 0.707 |
| Ensemble | 0.837 | 0.249 | 0.155 | 0.599 |
| GMM | 0.641 | 0.207 | −0.045 | 0.454 |
| RF | 0.804 | 0.179 | 0.013 | 0.632 |
| XGBoost | 0.804 | 0.178 | −0.045 | 0.661 |
| HMM | 0.674 | 0.162 | −0.080 | — |

### Phase 2 — ML vs DL vs Hybrid (68-sample test set)

| Approach | Accuracy | Macro F1 | Weighted F1 | MCC |
|----------|----------|----------|-------------|-----|
| A: ML Baseline (SVM) | 0.815 | 0.270 | 0.783 | 0.216 |
| B: DL Only (Fusion+cRT) | 0.603 | 0.182 | 0.686 | 0.002 |
| **C: Hybrid Weighted** | **0.926** | **0.507** | **0.905** | **0.520** |

**+87.5% relative improvement** in macro-F1 over the Phase 1 baseline.

**Note:** Initial Phase 1 results (XGBoost: 96.5%) were inflated by data
leakage. All results above use corrected split-before-augment protocols.

---

## References

1. Dunstan, P. (2006). *Dunstan Baby Language* — five universal cry categories.
2. Ji, C. et al. (2021). A review of infant cry analysis and classification. *EURASIP J. Audio Speech Music Process.*
3. Librosa documentation: https://librosa.org/doc/latest/

---

## License

This project is developed for academic purposes as part of a university
coursework submission.
