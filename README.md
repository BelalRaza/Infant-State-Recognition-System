# Infant Cry Classifier

Automatic classification of infant cry audio into five distress categories --
**hunger**, **belly pain**, **burping**, **discomfort**, and **tiredness** --
using signal-processing features and machine-learning models. The project is
structured in three phases: a classical ML baseline (Phase 1), a deep-learning
model (Phase 2), and a hybrid ensemble that fuses both approaches (Phase 3).

---

## Repository Structure

```
Infant-State-Recognition-System/
├── README.md
├── requirements.txt
├── setup.sh
│
├── pipelines/                          # Runnable entry points
│   ├── run_pipeline.py                 # Standard Phase 1 pipeline
│   ├── run_lightweight.py              # Memory-optimized variant
│   └── run_ultrasafe.py                # MacBook-safe variant
│
├── notebooks/                          # Colab notebooks
│   ├── Phase1_Classical_ML.ipynb       # Phase 1 classical ML pipeline
│   ├── Phase2_Deep_Learning.ipynb      # Phase 2 deep learning pipeline
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   └── 02_feature_engineering.ipynb    # Feature engineering exploration
│
├── src/                                # Source modules
│   ├── __init__.py
│   ├── config.py                       # Central configuration
│   ├── data_loader.py                  # Audio loading and splitting
│   ├── augmentation.py                 # 17 audio augmentation techniques
│   ├── feature_extractor.py            # 411-dim feature extraction
│   ├── feature_extraction.py           # 750-dim MFCC extraction (legacy)
│   ├── statistical_tests.py            # ADF, Shapiro-Wilk tests
│   ├── evaluation.py                   # Metrics, plots, model comparison
│   ├── gmm_classifier.py              # Gaussian Mixture Model
│   ├── svm_classifier.py              # SVM with SMOTE + OvO
│   ├── hmm_model.py                   # Hidden Markov Model
│   ├── rf_classifier.py               # Random Forest
│   ├── xgb_classifier.py              # XGBoost
│   ├── ensemble_classifier.py         # Stacking Ensemble
│   └── phase2/                         # Phase 2 deep learning modules
│       ├── config.py                   # DL hyperparameters
│       ├── models.py                   # CNN + BiLSTM + Attention architecture
│       ├── trainer.py                  # Training loop with LDAM + DRW
│       ├── losses.py                   # LDAM loss, DRW scheduler
│       ├── data_pipeline.py            # GroupShuffleSplit, augmentation
│       ├── features.py                 # Mel-spectrogram + domain features
│       ├── evaluation.py               # DL evaluation metrics
│       ├── hybrid.py                   # Hybrid weighted ensemble
│       ├── distillation.py             # Knowledge distillation + INT8
│       └── interpretability.py         # Grad-CAM, attention visualization
│
├── data/raw/                           # Original audio files by class
│   ├── hunger/     (382 files)
│   ├── discomfort/ (27 files)
│   ├── tiredness/  (24 files)
│   ├── belly_pain/ (16 files)
│   └── burping/    (8 files)
│
├── results/                            # All evaluation results
│   ├── phase1_leaky/                   # Pre-leakage-fix results
│   ├── phase1_corrected/               # Leak-free results (ground truth)
│   └── phase2/                         # Phase 2 DL + hybrid results
│
├── reports/                            # Reports and presentation
│   ├── phase1_report/                  # Phase 1 LaTeX report + PDF
│   ├── phase2_report/                  # Phase 2 LaTeX report + PDF
│   └── presentation/                   # HTML presentation + walkthrough
│
└── docs/                               # Documentation
    ├── EXAM_GUIDE.md                   # Comprehensive study guide
    ├── project_requirements.md         # Course requirements
    └── rubrics/                        # Grading rubrics
        ├── phase1_rubric.csv
        └── phase2_rubric.csv
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Infant-State-Recognition-System.git
cd Infant-State-Recognition-System

# 2. Create and activate a virtual environment (Python 3.10)
python3.10 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Phase 1 -- Classical ML Baseline

### Feature Pipeline

| Component | Detail |
|-----------|--------|
| Sample rate | Native 8,000 Hz (no upsampling) |
| Duration | Pad/truncate to 7 s (56,000 samples) |
| MFCC | 40 coefficients x 6 stats = 240 features |
| CQCC | 20 coefficients x 6 stats = 120 features |
| Pitch (F0) | pYIN algorithm, 7 statistics |
| Spectral contrast | 7 bands x 2 (peak + valley) = 14 features |
| Chroma | 12 bins x 2 stats = 24 features |
| Spectral descriptors | centroid, bandwidth, rolloff, flatness, ZCR, RMS = 6 |
| **Total** | **411 features per clip** |

### Models

| Model | Role | Library |
|-------|------|---------|
| **GMM** | Density-based generative classifier (one GMM per class) | `scikit-learn` |
| **SVM** | Discriminative classifier with SMOTE + OvO + grid search | `scikit-learn` |
| **HMM** | Left-right topology (8 states), temporal dynamics | `hmmlearn` |
| **Random Forest** | 500 trees with balanced subsampling | `scikit-learn` |
| **XGBoost** | Gradient boosting with class-weighted loss | `xgboost` |
| **Stacking Ensemble** | Meta-learner combining SVM + RF + XGBoost probabilities | `scikit-learn` |

---

## Phase 2 -- Deep Learning & Hybrid Ensemble

Phase 2 introduces a **CNN + BiLSTM + Temporal Attention** architecture on
64-band mel-spectrograms with LDAM loss, Deferred Re-Weighting, domain feature
fusion, and knowledge distillation (135K Teacher -> 16K Student). Data split
uses GroupShuffleSplit by infant UUID (70/15/15).

DL alone could not beat the Phase 1 SVM (best DL: Teacher macro-F1 = 0.293).
The breakthrough came from a **hybrid weighted ensemble** fusing SVM + RF + DL
probability outputs.

### Key Techniques

- LDAM loss with class-dependent margins
- Deferred Re-Weighting (uniform -> class-balanced at epoch 60)
- Mixup, SpecAugment, and balanced augmentation (326 -> 1,920 samples)
- Knowledge distillation + INT8 quantisation (35.9 KB, 11.4 ms inference)

---

## Phase 3 -- Edge Deployment *(planned)*

Phase 3 will deploy the INT8-quantised student model via TFLite Micro on
ESP32 and ONNX.js for browser-based inference.

---

## Results

### Phase 1 -- Classical ML (leak-free, 92-sample test set)

| Model | Accuracy | Macro F1 | MCC | AUC-ROC |
|-------|----------|----------|-----|---------|
| **SVM (SMOTE)** | 0.815 | **0.270** | 0.216 | 0.707 |
| Ensemble | 0.837 | 0.249 | 0.155 | 0.599 |
| GMM | 0.641 | 0.207 | -0.045 | 0.454 |
| RF | 0.804 | 0.179 | 0.013 | 0.632 |
| XGBoost | 0.804 | 0.178 | -0.045 | 0.661 |
| HMM | 0.674 | 0.162 | -0.080 | -- |

### Phase 2 -- ML vs DL vs Hybrid (68-sample test set)

| Approach | Accuracy | Macro F1 | Weighted F1 | MCC |
|----------|----------|----------|-------------|-----|
| A: ML Baseline (SVM) | 0.815 | 0.270 | 0.783 | 0.216 |
| B: DL Only (Fusion+cRT) | 0.603 | 0.182 | 0.686 | 0.002 |
| **C: Hybrid Weighted** | **0.926** | **0.507** | **0.905** | **0.520** |

**+87.5% relative improvement** in macro-F1 over the Phase 1 baseline.

### Data Leakage Note

Initial Phase 1 results (XGBoost: 96.5% accuracy) were inflated by data
leakage (augment-before-split). All results above use corrected
split-before-augment protocols with GroupShuffleSplit by infant UUID.

---

## Running the Pipeline

```bash
# Phase 1 -- standard pipeline (from project root)
python pipelines/run_pipeline.py

# Phase 1 -- memory-safe variant for MacBook
python pipelines/run_ultrasafe.py

# Phase 1 & 2 -- Colab notebooks (recommended)
# Upload notebooks/Phase1_Classical_ML.ipynb or
# notebooks/Phase2_Deep_Learning.ipynb to Google Colab
```

---

## References

1. Dunstan, P. (2006). *Dunstan Baby Language* -- five universal cry categories.
2. Ji, C. et al. (2021). A review of infant cry analysis and classification. *EURASIP J. Audio Speech Music Process.*
3. Cao, K. et al. (2019). Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. *NeurIPS 2019.*
4. Librosa documentation: https://librosa.org/doc/latest/

---

## License

This project is developed for academic purposes as part of a university
coursework submission.
