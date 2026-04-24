<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0F766E&height=220&section=header&text=Infant%20State%20Recognition&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=30&desc=Audio%20Classification%20System&descSize=16&descAlignY=52&descColor=2DD4BF" />

<br>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com/?lines=Decoding+infant+cries+through+audio+intelligence;411+acoustic+features+%C2%B7+6+ML+models+%C2%B7+CNN%2BBiLSTM;Hybrid+ensemble+%E2%80%94+87%25+improvement+over+baseline;Edge+ready+%E2%80%94+35.9+KB+INT8+quantized+model&font=Fira+Code&center=true&width=680&height=35&color=2DD4BF&vCenter=true&pause=1200&size=14&duration=3500" />
</a>

<br>
<br>

![](https://img.shields.io/badge/Python_3.10-0F766E?style=flat-square)&nbsp;&nbsp;
![](https://img.shields.io/badge/Phase_2_Complete-0F766E?style=flat-square)&nbsp;&nbsp;
![](https://img.shields.io/badge/457_Audio_Samples-0F766E?style=flat-square)&nbsp;&nbsp;
![](https://img.shields.io/badge/5_Cry_Categories-0F766E?style=flat-square)&nbsp;&nbsp;
![](https://img.shields.io/badge/Academic_Project-0F766E?style=flat-square)

<br>
<br>

**Automatic classification of infant cry audio into five distress categories using a three-phase**
**approach: classical machine learning, deep learning, and hybrid ensemble fusion.**
**Designed to work with extreme class imbalance (47:1) and deployable at the edge in 35.9 KB.**

<br>

</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

## <img src="https://img.shields.io/badge/01-0F766E?style=flat-square" height="24" /> &nbsp; The Challenge

Infant cries encode critical information about a baby's needs, but distinguishing between cry types is challenging — even for experienced caregivers. This project tackles the problem computationally, classifying cries into five categories defined by the **Dunstan Baby Language** framework.

<br>

<div align="center">

| Category | Samples | Share |
|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />-:|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />--:|
| Hunger | 382 | 83.6% |
| Discomfort | 27 | 5.9% |
| Tiredness | 24 | 5.2% |
| Belly Pain | 16 | 3.5% |
| Burping | 8 | 1.7% |

</div>

<br>

> With a **47:1 imbalance ratio** between the majority and minority classes, the core challenge is building models that recognize rare cry types without being overwhelmed by the dominant hunger class. All evaluation uses **macro-F1** to ensure minority class performance is not hidden by majority class accuracy.

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

## <img src="https://img.shields.io/badge/02-0F766E?style=flat-square" height="24" /> &nbsp; Methodology

A three-phase approach that progressively builds from classical signal processing to neural architectures, culminating in a hybrid system that outperforms either approach alone.

<br>

<div align="center">

| Phase | Approach | Input | Best Macro-F1 |
|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />--:|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />-:|
| **1** | Classical ML — 411 features, 6 models | Handcrafted acoustic features | 0.270 |
| **2** | Deep Learning — CNN + BiLSTM + Attention | 64-band mel-spectrogram | 0.293 |
| **2** | **Hybrid Ensemble — ML + DL fusion** | Probability outputs | **0.507** |
| **3** | Edge Deployment — INT8 quantization | Distilled student model | _planned_ |

</div>

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

## <img src="https://img.shields.io/badge/03-0F766E?style=flat-square" height="24" /> &nbsp; Feature Engineering

The Phase 1 pipeline extracts **411 acoustic features** per audio clip, capturing spectral, temporal, and tonal characteristics of infant cries.

<br>

<div align="center">

| Component | Method | Dimensions |
|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />-|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />-|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />-:|
| MFCC | 40 coefficients x 6 statistics | 240 |
| CQCC | 20 coefficients x 6 statistics | 120 |
| Pitch (F0) | pYIN algorithm, 7 statistics | 7 |
| Spectral Contrast | 7 bands x 2 (peak + valley) | 14 |
| Chroma | 12 bins x 2 statistics | 24 |
| Spectral Descriptors | centroid, bandwidth, rolloff, flatness, ZCR, RMS | 6 |
| | **Total** | **411** |

</div>

<br>

All audio is processed at native **8,000 Hz** (no upsampling) and normalized to **7-second** windows via padding or truncation.

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

## <img src="https://img.shields.io/badge/04-0F766E?style=flat-square" height="24" /> &nbsp; Models

### Phase 1 — Classical ML

<br>

<div align="center">

| Model | Strategy | Library |
|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />-|
| GMM | Density-based generative (one GMM per class) | scikit-learn |
| SVM | SMOTE + One-vs-One + grid search | scikit-learn |
| HMM | Left-right topology, 8 states | hmmlearn |
| Random Forest | 500 trees, balanced subsampling | scikit-learn |
| XGBoost | Gradient boosting, class-weighted loss | xgboost |
| Stacking Ensemble | Meta-learner over SVM + RF + XGBoost | scikit-learn |

</div>

<br>

### Phase 2 — Deep Learning

The deep learning pipeline introduces a **CNN + BiLSTM + Temporal Attention** architecture trained on 64-band mel-spectrograms with domain feature fusion.

<br>

**Key techniques:**

- **LDAM loss** with class-dependent margins for imbalanced learning
- **Deferred Re-Weighting** — uniform loss weighting transitions to class-balanced at epoch 60
- **Mixup + SpecAugment** augmentation pipeline (326 → 1,920 training samples)
- **Knowledge distillation** — 135K-parameter Teacher → 16K-parameter Student
- **INT8 quantization** — 35.9 KB model with 11.4 ms inference latency
- **GroupShuffleSplit** by infant UUID to prevent data leakage (70/15/15 split)

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

## <img src="https://img.shields.io/badge/05-0F766E?style=flat-square" height="24" /> &nbsp; Results

### Phase 1 — Classical ML Baseline

<sub>92-sample test set, leak-free evaluation</sub>

<br>

<div align="center">

| Model | Accuracy | Macro F1 | MCC | AUC-ROC |
|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />--:|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />--:|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />:|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />-:|
| **SVM (SMOTE)** | 0.815 | **0.270** | 0.216 | 0.707 |
| Stacking Ensemble | 0.837 | 0.249 | 0.155 | 0.599 |
| GMM | 0.641 | 0.207 | -0.045 | 0.454 |
| Random Forest | 0.804 | 0.179 | 0.013 | 0.632 |
| XGBoost | 0.804 | 0.178 | -0.045 | 0.661 |
| HMM | 0.674 | 0.162 | -0.080 | — |

</div>

<br>

### Phase 2 — ML vs DL vs Hybrid

<sub>68-sample test set, GroupShuffleSplit by infant UUID</sub>

<br>

<div align="center">

| Approach | Accuracy | Macro F1 | Weighted F1 | MCC |
|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />--:|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />--:|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" /><img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />--:|:<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />:|
| ML Baseline (SVM) | 0.815 | 0.270 | 0.783 | 0.216 |
| DL Only (Fusion + cRT) | 0.603 | 0.182 | 0.686 | 0.002 |
| **Hybrid Weighted** | **0.926** | **0.507** | **0.905** | **0.520** |

</div>

<br>

> **+87.5% relative improvement** in macro-F1 over the Phase 1 baseline. The hybrid weighted ensemble fuses SVM + Random Forest + DL probability outputs, achieving what neither classical ML nor deep learning could alone.

<br>

<details>
<summary><b>Data Leakage Note</b></summary>
<br>

Initial Phase 1 results (XGBoost: 96.5% accuracy) were inflated by data leakage — augmentation was applied before the train/test split, allowing augmented copies of test samples to appear in training. All results above use corrected **split-before-augment** protocols with GroupShuffleSplit by infant UUID.

</details>

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

## <img src="https://img.shields.io/badge/06-0F766E?style=flat-square" height="24" /> &nbsp; Tech Stack

<br>

<div align="center">

![Python](https://img.shields.io/badge/python-0F766E?style=for-the-badge&logo=python&logoColor=ffffff)&nbsp;
![PyTorch](https://img.shields.io/badge/pytorch-0F766E?style=for-the-badge&logo=pytorch&logoColor=ffffff)&nbsp;
![scikit-learn](https://img.shields.io/badge/scikit--learn-0F766E?style=for-the-badge&logo=scikitlearn&logoColor=ffffff)&nbsp;
![NumPy](https://img.shields.io/badge/numpy-0F766E?style=for-the-badge&logo=numpy&logoColor=ffffff)&nbsp;
![Pandas](https://img.shields.io/badge/pandas-0F766E?style=for-the-badge&logo=pandas&logoColor=ffffff)

![Jupyter](https://img.shields.io/badge/jupyter-0F766E?style=for-the-badge&logo=jupyter&logoColor=ffffff)&nbsp;
![Google Colab](https://img.shields.io/badge/google_colab-0F766E?style=for-the-badge&logo=googlecolab&logoColor=ffffff)&nbsp;
![XGBoost](https://img.shields.io/badge/xgboost-0F766E?style=for-the-badge)&nbsp;
![Librosa](https://img.shields.io/badge/librosa-0F766E?style=for-the-badge)&nbsp;
![Matplotlib](https://img.shields.io/badge/matplotlib-0F766E?style=for-the-badge)

</div>

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

<details>
<summary><h2><img src="https://img.shields.io/badge/07-0F766E?style=flat-square" height="24" /> &nbsp; Project Structure</h2></summary>
<br>

```
Infant-State-Recognition-System/
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
│   ├── config.py                       # Central configuration
│   ├── data_loader.py                  # Audio loading and splitting
│   ├── augmentation.py                 # 17 audio augmentation techniques
│   ├── feature_extractor.py            # 411-dim feature extraction
│   ├── evaluation.py                   # Metrics, plots, model comparison
│   ├── gmm_classifier.py              # Gaussian Mixture Model
│   ├── svm_classifier.py              # SVM with SMOTE + OvO
│   ├── hmm_model.py                   # Hidden Markov Model
│   ├── rf_classifier.py               # Random Forest
│   ├── xgb_classifier.py              # XGBoost
│   ├── ensemble_classifier.py         # Stacking Ensemble
│   └── phase2/                         # Phase 2 deep learning modules
│       ├── config.py                   # DL hyperparameters
│       ├── models.py                   # CNN + BiLSTM + Attention
│       ├── trainer.py                  # Training loop (LDAM + DRW)
│       ├── losses.py                   # LDAM loss, DRW scheduler
│       ├── data_pipeline.py            # GroupShuffleSplit, augmentation
│       ├── features.py                 # Mel-spectrogram + domain features
│       ├── evaluation.py               # DL evaluation metrics
│       ├── hybrid.py                   # Hybrid weighted ensemble
│       ├── distillation.py             # Knowledge distillation + INT8
│       └── interpretability.py         # Grad-CAM, attention maps
│
├── data/raw/                           # Original audio files by class
│   ├── hunger/          382 files
│   ├── discomfort/       27 files
│   ├── tiredness/        24 files
│   ├── belly_pain/       16 files
│   └── burping/           8 files
│
├── results/                            # Evaluation outputs
│   ├── phase1_corrected/               # Leak-free Phase 1 results
│   └── phase2/                         # Phase 2 DL + hybrid results
│
├── reports/                            # LaTeX reports and presentation
│   ├── phase1_report/
│   ├── phase2_report/
│   └── presentation/
│
├── docs/                               # Documentation and rubrics
├── requirements.txt
└── setup.sh
```

</details>

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

## <img src="https://img.shields.io/badge/08-0F766E?style=flat-square" height="24" /> &nbsp; Getting Started

```bash
# Clone
git clone https://github.com/BelalRaza/Infant-State-Recognition-System.git
cd Infant-State-Recognition-System

# Environment
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

<br>

**Run locally:**

```bash
# Phase 1 — standard pipeline
python pipelines/run_pipeline.py

# Phase 1 — memory-safe variant for MacBook
python pipelines/run_ultrasafe.py
```

**Run on Colab** _(recommended)_:

Upload `notebooks/Phase1_Classical_ML.ipynb` or `notebooks/Phase2_Deep_Learning.ipynb` to [Google Colab](https://colab.research.google.com).

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

## <img src="https://img.shields.io/badge/09-0F766E?style=flat-square" height="24" /> &nbsp; References

1. Dunstan, P. (2006). *Dunstan Baby Language* — five universal cry categories.
2. Ji, C. et al. (2021). A review of infant cry analysis and classification. *EURASIP Journal on Audio, Speech, and Music Processing.*
3. Cao, K. et al. (2019). Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. *NeurIPS 2019.*
4. McFee, B. et al. *Librosa: Audio and music signal analysis in Python.* [librosa.org](https://librosa.org/doc/latest/)

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0F766E&height=2" />

<br>

<div align="center">

<sub>Developed for academic coursework — University project on infant state recognition through audio analysis.</sub>

<br>
<br>

</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0F766E&height=120&section=footer" />
