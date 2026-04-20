# EXAM-READY GUIDE: Infant State Recognition System

> **Read time: ~15 minutes. You will be able to answer ANY question a professor asks.**

---

## 1. THE BIG PICTURE (What is this project?)

**Problem statement:** Newborn babies cry for different reasons (hunger, pain, tiredness, etc.), but parents and nurses cannot always tell WHY. We built a **machine learning system that LISTENS to a baby's cry audio and TELLS you the reason** behind it.

**Why it matters:**
- New parents lose sleep trying to figure out why their baby is crying
- In **NICUs** (Neonatal Intensive Care Units), nurses monitor dozens of babies -- automated cry analysis saves lives
- Faster response = less stress on the infant and caregiver

**What we classify -- 5 categories:**

| Class | What it means |
|-------|--------------|
| **hunger** | Baby needs to be fed |
| **belly_pain** | Baby has stomach/gas pain |
| **burping** | Baby needs to burp (trapped air) |
| **discomfort** | Baby is uncomfortable (wet diaper, too hot/cold) |
| **tiredness** | Baby is sleepy and overtired |

**Analogy:** Think of it like **Shazam, but for baby cries**. Shazam listens to music and tells you the song name. Our system listens to a baby cry and tells you the reason.

**Phase 1 = Classical ML only** (this project). Phase 2 = Deep Learning. Phase 3 = Hybrid + deployment.

---

## 2. THE SCIENCE BEHIND IT

**Babies cannot talk, but their cries ARE different depending on what is wrong.** This is NOT guesswork -- it is backed by decades of research.

### Dunstan Baby Language
An Australian woman named Priscilla Dunstan identified **5 reflexive cry sounds** that all babies make:

| Sound | Meaning | Physical Reflex |
|-------|---------|----------------|
| **Neh** | Hunger | Sucking reflex pushes tongue to roof of mouth |
| **Eairh** | Belly pain / gas | Lower abdomen muscles tighten |
| **Eh** | Burping | Chest rises trying to release trapped air |
| **Heh** | Discomfort | Skin-based stress response |
| **Owh** | Tiredness | Yawning reflex shapes the mouth into an O |

### Brain Science
- The **PAG (Periaqueductal Gray)** is the brain region that controls crying in mammals
- It triggers different **phonation modes** (ways of producing voice):
  - **Modal voice** = normal crying (regular vibration of vocal cords)
  - **Hyperphonation** = very high pitch, above 1000 Hz (indicates **pain**)
  - **Dysphonation** = harsh, rough, noisy sound (indicates **distress**)

### Research History
- This is NOT made up. Research on infant cry analysis goes back to **1968 (Wasz-Hockert et al.)**
- Peer-reviewed studies show acoustic differences between cry types are measurable and classifiable

---

## 3. THE DATASET

- **Source:** Donate-a-Cry corpus (open-source infant cry recordings)
- **Original count:** ~457 audio files (**.wav format** = raw audio waveforms)
- **Sample rate:** 8000 Hz (8000 measurements of the sound wave per second)
- **Clip length:** 7 seconds each (shorter clips are **padded with silence**, longer ones are **cut**)

### The HUGE Problem: Class Imbalance

| Class | Original Count |
|-------|---------------|
| hunger | 382 |
| belly_pain | ~16 |
| burping | ~8 |
| discomfort | ~18 |
| tiredness | ~33 |

**382 hunger files but only 8 burping files!** A model would just guess "hunger" every time and be right 80%+ of the time. This is useless.

### Solution: Data Augmentation (see Section 5)

**After augmentation:**

| Class | Final Count |
|-------|------------|
| hunger | 382 (untouched -- already the largest) |
| belly_pain | 288 |
| burping | 144 |
| discomfort | 300 |
| tiredness | 300 |
| **Total** | **1414 files** |

### Train/Test Split
- **80% train** (1131 clips) / **20% test** (283 clips)
- **Stratified split:** preserves class proportions in both sets (so every class appears in both train AND test)
- **Random seed = 42** for reproducibility

---

## 4. WHAT IS AUDIO DATA? (Explain Like I'm 5)

1. **Sound** = vibrations in the air
2. A **microphone** converts those vibrations into electrical signals, which become **numbers**
3. A **WAV file** = a list of numbers representing the **amplitude** (loudness) at each tiny moment in time
4. **Sample rate 8000 Hz** = we record **8000 numbers per second**
5. Each clip is 7 seconds, so: **7 x 8000 = 56,000 numbers per audio clip**
6. So each baby cry is literally just an **array of 56,000 numbers**

**But you CANNOT feed 56,000 raw numbers to a model.** That is too much noise and not enough signal. You need to extract **meaningful patterns** -- that is what Feature Engineering does (Section 6).

---

## 5. DATA AUGMENTATION (Making More Training Data)

**Problem:** Only 8 burping files! Models need hundreds of examples to learn.

**Solution:** Take existing audio and **modify it slightly** to create "new" samples.

**Analogy:** Like taking a photo and flipping, rotating, or brightening it. The photo still shows the same thing, but it looks slightly different, giving the model more examples to learn from.

### 17 Augmentation Techniques Used:

| Technique | What it does | Suffix |
|-----------|-------------|--------|
| **Pitch shift** (+1, -1, +2, -2, +0.5, -0.5 semitones) | Makes voice higher or lower | ps+1, ps-1, etc. |
| **Time stretch** (0.85x, 0.9x, 1.1x, 1.15x) | Makes it faster/slower WITHOUT changing pitch | ts085, ts09, etc. |
| **White noise** | Adds background static (SNR ~25-30 dB) | wn |
| **Time shift** (+/- 0.5 seconds) | Shifts audio left/right, padding with silence | sh+, sh- |
| **Combination** | Pitch shift +1 semitone AND add light noise | combo |
| **Random gain** | Makes louder or quieter (random within +/-6 dB) | gain |
| **Band-pass filter** | Removes very high/low frequencies (simulates different microphones) | bp |
| **Reverb simulation** | Adds echo effect (simulates different room acoustics) | reverb |

**Target:** At least 300 samples per class. Hunger was already at 382, so it was **left alone** (never augment the dominant class -- that makes imbalance worse).

Augmented files are saved with `_aug_` in the filename so they can be identified.

---

## 6. FEATURE ENGINEERING (The Most Important Part!)

> **"We cannot give raw audio to a model. We need to DESCRIBE the audio using numbers that capture its CHARACTER."**

**Analogy:** Think of describing a person's voice to someone: "deep, raspy, loud, fast-talking." Those descriptive words are **features**. We do the same thing, but with precise numbers.

**We extract 411 numbers (features) from each 7-second audio clip.**

---

### a) MFCC -- 240 features -- "What the sound SOUNDS like to human ears"

- **MFCC = Mel-Frequency Cepstral Coefficients**
- Simulates how **human ears** perceive sound
- Human ears are better at distinguishing low-pitched sounds than high-pitched ones
- The **Mel scale** converts raw frequency (Hz) to how humans actually perceive pitch

**Process:** Audio --> **FFT** (break into frequencies) --> **Mel filter bank** (weight by human hearing) --> **Log** (compress dynamic range) --> **DCT** (decorrelate) --> MFCCs

- We extract **40 MFCCs per time frame**, then compute **mean and standard deviation** across all frames
- We also compute **delta** (rate of change over time) and **delta-delta** (acceleration of change)
- **Total: 40 coefficients x 2 stats x 3 types (MFCC + delta + delta2) = 240**

**Analogy:** Like describing a voice as "this average pitch, varying this much, getting higher/lower over time."

---

### b) CQCC -- 120 features -- "Better version for baby cries"

- **CQCC = Constant-Q Cepstral Coefficients**
- This is a **KEY INNOVATION** of our project

**Problem with MFCC:** It uses fixed-size frequency analysis windows (same resolution everywhere).

**CQT (Constant-Q Transform):** Uses **variable-size windows**:
- **Low frequencies** --> high frequency resolution (good for detecting the **fundamental pitch F0**)
- **High frequencies** --> high time resolution (good for detecting **cry onset and offset**)

Baby cries have strong **harmonic patterns** that CQT captures better than standard FFT.

- **20 coefficients x 2 stats x 3 types (CQCC + delta + delta2) = 120**

**Analogy:** "MFCC is like a ruler with equal spacing. CQCC is like a ruler where markings are closer together at one end -- better for measuring the specific patterns in baby cries."

---

### c) Pitch/F0 Features -- 7 features -- "How HIGH or LOW the cry is"

- **F0 = fundamental frequency** = the base pitch of the voice
- **Pain cries:** very high pitch (hyperphonation, > 1000 Hz)
- **Tiredness cries:** lower, more monotone
- **7 features:** f0_mean, f0_std, f0_max, f0_min, f0_range, voiced_fraction, f0_slope

**Analogy:** "Is the baby screaming HIGH (pain) or moaning LOW (tired)?"

*Note: In the memory-safe pipeline, F0 is estimated from spectral centroid rather than the pyin algorithm, to avoid memory issues on MacBook.*

---

### d) Spectral Contrast -- 14 features -- "Difference between loud and quiet frequencies"

- Measures the **peak-to-valley difference** in the frequency spectrum across sub-bands
- Different cry types have different harmonic structures
- **7 bands x 2 stats (mean, std) = 14**

**Analogy:** "Like measuring how 'textured' or 'rich' the sound is -- smooth hum vs. rough scream."

---

### e) Chroma -- 24 features -- "Musical note distribution"

- Maps ALL frequencies to the **12 musical notes** (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
- Shows which musical notes are present in the cry
- Different cry types emphasize different notes
- **12 bins x 2 stats (mean, std) = 24**

**Analogy:** "What musical notes is the baby hitting when it cries?"

---

### f) Spectral Descriptors -- 6 features -- "Overall sound character"

| Feature | What it measures |
|---------|-----------------|
| **Zero Crossing Rate** | How often the signal crosses zero (noisiness) |
| **Spectral Centroid** | "Center of gravity" of frequency (brightness) |
| **Spectral Rolloff** | Frequency below which 85% of energy sits |
| **Spectral Bandwidth** | Spread/width of frequencies |
| **RMS Energy** | Overall loudness |
| **Spectral Flatness** | How noise-like vs tone-like the sound is |

---

### TOTAL: 240 + 120 + 7 + 14 + 24 + 6 = 411 features per audio clip

So each 7-second cry clip becomes a single row of **411 numbers** that describe everything about that cry's acoustic character.

---

## 7. ALL MODELS EXPLAINED (Simple Analogies)

---

### a) GMM (Gaussian Mixture Model) -- Accuracy: 62.2%

**Simple idea:** Each class has a "shape" in feature space, like a cloud. GMM models each class as a **mixture of bell curves (Gaussians)**.

- For each class, it learns: "hunger cries form a cloud centered HERE with THIS spread"
- **Prediction:** Which class's cloud does this new cry fit into best?
- We use **Bayesian GMM** -- automatically figures out how many bell curves (components) are needed
- **Dirichlet process:** starts with 8 components, prunes unnecessary ones
- **One GMM per class** (one-vs-all approach)
- **Covariance type: diagonal** (assumes features are independent within each component)
- Features are **StandardScaler** normalized before fitting
- **Class-weighted log-priors** handle imbalance

**Why lowest accuracy (62.2%)?** Density estimation in **411 dimensions** is extremely hard -- this is the "**curse of dimensionality**." There simply are not enough samples to reliably estimate the shape of a 411-dimensional cloud.

**Analogy:** "Like trying to describe the shape of a crowd of people using overlapping circles -- in 411-dimensional space, you would need millions of people to get the shapes right."

---

### b) SVM (Support Vector Machine) -- Accuracy: 92.6%

**Simple idea:** Draw the **BEST possible dividing boundary** between classes.

- In 2D: draw a line. In 411D: draw a **hyperplane**
- **RBF kernel:** can draw **CURVED** boundaries (not just straight lines). It projects data into an even higher dimensional space where classes become linearly separable.
- **SMOTE** (Synthetic Minority Over-sampling): creates **synthetic minority samples** before training. Like drawing a line between two existing burping samples and creating a new point along that line.
- **One-vs-One (OvO):** for 5 classes, trains **10 separate SVMs** (one for each pair of classes: hunger-vs-pain, hunger-vs-burping, etc.)
- **class_weight='balanced'** gives more penalty for misclassifying rare classes
- **C=10** (regularization strength), **gamma='scale'**
- Pipeline: StandardScaler --> SMOTE --> SVC

**Why good (92.6%)?** SVMs are excellent at finding optimal boundaries in high dimensions. They only care about the **support vectors** (the hardest-to-classify points near the boundary).

**Analogy:** "Like drawing the widest possible road between two neighborhoods -- the wider the road, the less likely you are to accidentally cross into the wrong neighborhood."

---

### c) Random Forest (RF) -- Accuracy: 92.9%

**Simple idea:** Ask **200 decision trees** yes/no questions, then **majority vote** wins.

- Each tree asks questions like: "Is MFCC_3_mean > 0.5? Is f0_range > 200?"
- Each tree only sees a **random subset** of features and data (**bagging**)
- **max_features='sqrt':** each tree considers only sqrt(411) ~ 20 features at each split
- **class_weight='balanced_subsample':** gives more weight to rare classes within each tree's bootstrap sample
- **min_samples_leaf=2:** prevents trees from having leaves with just 1 sample
- **n_estimators=200** (200 trees in the pipeline run; defined as 500 in the class defaults)
- Final answer: **vote among all trees**

**Why good (92.9%)?** Robust, very hard to overfit, works well with many features, and naturally provides **feature importance rankings**.

**Analogy:** "Like asking 200 different doctors, each with partial information about the patient, and going with the majority opinion."

---

### d) XGBoost (Extreme Gradient Boosting) -- Accuracy: 96.5% (BEST!)

**Simple idea:** Build trees **ONE AT A TIME**, each one **fixing the MISTAKES** of all previous trees.

- Tree 1 makes predictions
- Tree 2 focuses on what Tree 1 got WRONG
- Tree 3 focuses on what is STILL wrong
- ... continue for **150 trees** (n_estimators=150 in pipeline)

**Key hyperparameters:**
- **max_depth=6:** each tree can ask up to 6 levels of questions
- **learning_rate=0.1:** each tree only makes a small correction (prevents overfitting)
- **subsample=0.8:** each tree sees 80% of training data
- **colsample_bytree=0.8:** each tree sees 80% of features
- **Balanced sample weights** via `compute_sample_weight("balanced", y)`
- **Objective: multi:softprob** (outputs probabilities for all 5 classes)

**Why BEST (96.5%)?** Iterative error correction + regularization = highly accurate. XGBoost is the **gold standard for tabular/feature-based data**.

**Analogy:** "Like a student doing 150 practice exams, each time focusing SPECIFICALLY on the questions they got wrong last time. By the end, they have mastered almost everything."

---

### e) Ensemble (Stacking) -- Accuracy: 95.1%

**Simple idea:** Combine **SVM + RF + XGBoost** and let a "judge" decide how much to trust each one.

- **Step 1:** Get **probability predictions** from all 3 base models (each outputs 5 probabilities = 15 numbers total)
- **Step 2:** Feed those 15 numbers into a **Logistic Regression meta-learner**
- The meta-learner learns **WHEN to trust which model**
- **class_weight='balanced'**, **multinomial** classification
- **Highest AUC-ROC (0.994)** = best at ranking how confident it is

**Why slightly lower accuracy than XGBoost alone?** XGBoost is already so good that averaging with slightly weaker models (SVM, RF) **dilutes** its performance. But Ensemble has the **best probability calibration** (AUC-ROC 0.994).

**Analogy:** "Like having 3 expert judges and a head judge who decides how much to trust each one based on their track records."

---

## 8. WHY THESE MODELS? WHY NOT OTHERS?

| Model | Why chosen |
|-------|-----------|
| **GMM** | Classic probabilistic model for audio -- used since 1990s for speaker recognition and speech |
| **SVM** | Gold standard for MFCC-based classification -- widely published in audio/speech literature |
| **Random Forest** | Robust baseline, handles imbalanced data well, provides interpretability via feature importance |
| **XGBoost** | State-of-the-art for tabular/feature-based data -- consistently wins Kaggle competitions |
| **Ensemble** | Standard technique to combine multiple models -- often improves overall robustness |

**Why NOT neural networks?**
- **Phase 1 = Classical ML only**
- **Phase 2** will use deep learning: CNN on mel spectrograms, LSTM for temporal patterns, Wav2Vec 2.0 transfer learning
- **Phase 3** will fuse classical + deep learning and deploy on mobile/edge devices

**HMM (Hidden Markov Model)** was implemented but **skipped** during the pipeline run due to memory issues on MacBook. It uses frame-level MFCC sequences (temporal modeling) rather than clip-level feature vectors.

---

## 9. RESULTS COMPARISON

### Full Results Table

| Model | Accuracy | Macro F1 | Weighted F1 | MCC | Kappa | AUC-ROC |
|-------|----------|----------|-------------|-----|-------|---------|
| **GMM** | 62.2% | 0.547 | 0.599 | 0.519 | 0.506 | 0.801 |
| **SVM** | 92.6% | 0.921 | 0.925 | 0.906 | 0.905 | 0.989 |
| **RF** | 92.9% | 0.929 | 0.929 | 0.910 | 0.910 | 0.987 |
| **XGBoost** | **96.5%** | **0.965** | **0.965** | **0.955** | **0.955** | 0.992 |
| **Ensemble** | 95.1% | 0.944 | 0.950 | 0.937 | 0.937 | **0.994** |

**Winner: XGBoost** (highest accuracy, F1, MCC, Kappa). **Ensemble** has the highest AUC-ROC.

### XGBoost Per-Class Breakdown (on 283 test samples)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| hunger | 0.93 | 0.99 | 0.96 | 76 |
| belly_pain | 0.98 | 0.98 | 0.98 | 58 |
| burping | 0.97 | 0.97 | 0.97 | 29 |
| discomfort | 0.97 | 0.95 | 0.96 | 60 |
| tiredness | 1.00 | 0.93 | 0.97 | 60 |

**Key observations:**
- **belly_pain** is easiest to classify (F1=0.98) -- pain cries are acoustically very distinct (high pitch)
- **tiredness** has perfect precision (1.00) but slightly lower recall (0.93) -- when the model says "tired," it is always right, but it misses some tired cries
- **hunger** has the highest recall (0.99) -- the model catches almost every hunger cry

### GMM Failure Analysis
GMM only got **6.9% recall on burping** (caught 2 out of 29). It classified most burping cries as hunger or discomfort. The 411-dimensional feature space is simply too high for density estimation with limited data.

---

## 10. METRICS EXPLAINED (What Do All These Numbers Mean?)

### Accuracy
- **Simplest metric:** percentage of correct predictions out of total
- **96.5% = out of 283 test samples, ~273 were correct**
- Limitation: can be misleading with imbalanced classes

### Precision
- **"Of everything the model SAID was hunger, how many actually were?"**
- High precision = **few false alarms**
- Formula: TP / (TP + FP)
- **Analogy:** "If I point at 100 babies and say 'that one is hungry,' precision = how many I was right about"

### Recall (Sensitivity)
- **"Of ALL the actual hunger samples, how many did the model CATCH?"**
- High recall = **few misses**
- Formula: TP / (TP + FN)
- **Analogy:** "Out of 76 real hunger cries in the test set, recall = how many I correctly identified"

### F1 Score
- **Harmonic mean** of Precision and Recall: **F1 = 2 x (P x R) / (P + R)**
- Why harmonic mean? It **penalizes** when either P or R is low. You cannot cheat by having high P and low R.
- **Macro F1:** Average F1 across all 5 classes (treats each class **equally** regardless of size)
- **Weighted F1:** Average weighted by the **number of samples** per class

### MCC (Matthews Correlation Coefficient)
- Like accuracy but **works even with imbalanced classes**
- Range: **-1** (all wrong) to **+1** (all correct), **0** = random guessing
- **0.955 = extremely strong correlation between predictions and truth**

### Cohen's Kappa
- Measures how much **better than random guessing** the model is
- Accounts for agreement that would happen by chance
- **0.955 = almost perfect agreement beyond chance**
- Interpretation: < 0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, > 0.8 almost perfect

### AUC-ROC
- **Area Under the Receiver Operating Characteristic Curve**
- Measures how well the model **RANKS** its predictions (probability quality)
- **1.0 = perfect ranking**, **0.5 = random**
- Ensemble's **0.994 = nearly perfect probability estimates**
- **Why it matters:** In medical applications (NICU monitoring), you want good **probability estimates**, not just hard yes/no classifications. A nurse needs to know "90% confidence this is pain" vs "51% confidence."
- Computed using **One-vs-Rest (OvR)** macro averaging across all 5 classes

---

## 11. GRAPHS AND VISUALIZATIONS EXPLAINED

All plots are saved in `results/plots/`.

### Confusion Matrix (e.g., `XGBoost_confusion_matrix.png`)
- A **grid/table** showing predicted class vs actual class
- **Rows** = what the baby ACTUALLY was
- **Columns** = what the model PREDICTED
- **Diagonal cells** = correct predictions (darker blue = more correct)
- **Off-diagonal cells** = mistakes (which classes get confused with each other)
- **Normalized version:** each row sums to 1.0 (shows percentages instead of raw counts)
- **What to look for:** strong dark diagonal = good model; scattered off-diagonal = model struggles

### t-SNE Projection (`tsne_feature_space.png`)
- **t-SNE = t-distributed Stochastic Neighbor Embedding**
- Takes the **411 features** and squashes them into **2D** for visualization
- Each dot = one audio clip, **color = class**
- If clusters are **well-separated** --> the features are discriminative (good!)
- If clusters **overlap** --> those classes are hard to distinguish
- Shows that our 411-dim features DO separate the 5 cry classes

### Model Comparison Bar Chart (`model_comparison_macro_f1.png`)
- Bar chart: **Macro F1** for each model side by side
- Visual confirmation: XGBoost > RF > SVM >> GMM

### Feature Importance Plots (`RF_feature_importance.png`, `XGBoost_feature_importance.png`)
- Shows the **top 20 most important features** for classification
- Tells us which acoustic properties best distinguish cry types
- Typically top features: **MFCC and CQCC coefficients** (they capture the most discriminative acoustic patterns)

### Class Distribution (`class_distribution.png`)
- Bar chart showing number of samples per class
- Confirms augmentation worked

### Per-Class F1 Charts (e.g., `XGBoost_f1_per_class.png`)
- Bar chart of F1 score for each class within one model
- Shows which cry types are easy vs hard to classify

### Additional Plots
- `waveforms_per_class.png` -- raw audio waveforms for each class
- `spectrograms_per_class.png` -- frequency-over-time heatmaps
- `mfcc_heatmaps.png` -- visual representation of MFCC features
- `feature_correlation_heatmap.png` -- which features are correlated with each other
- `augmentation_demo.png` -- before/after augmentation examples

---

## 12. CODEBASE STRUCTURE

```
Infant-State-Recognition-System/
├── data/
│   └── raw/                      # Audio files organized by class
│       ├── hunger/               # 382 WAV files
│       ├── belly_pain/           # 288 WAV files (after augmentation)
│       ├── burping/              # 144 WAV files (after augmentation)
│       ├── discomfort/           # 300 WAV files (after augmentation)
│       └── tiredness/            # 300 WAV files (after augmentation)
├── src/                          # All source code modules
│   ├── config.py                 # Central settings (sample rate, paths, class names)
│   ├── data_loader.py            # Loads WAV files into numpy arrays
│   ├── augmentation.py           # 17 augmentation techniques
│   ├── feature_extractor.py      # 411-dim feature extraction (MFCC+CQCC+pitch+...)
│   ├── feature_extraction.py     # Older 750-dim MFCC-only extractor (legacy)
│   ├── gmm_classifier.py         # Bayesian GMM (one per class)
│   ├── svm_classifier.py         # SVM with SMOTE + One-vs-One
│   ├── hmm_model.py              # Hidden Markov Model (skipped -- memory issues)
│   ├── rf_classifier.py          # Random Forest (200 trees)
│   ├── xgb_classifier.py         # XGBoost (150 rounds)
│   ├── ensemble_classifier.py    # Stacking ensemble (SVM+RF+XGBoost -> LogReg)
│   └── evaluation.py             # Metrics, plots, confusion matrices
├── run_ultrasafe.py              # Main pipeline script (memory-safe, used for results)
├── run_pipeline.py               # Full pipeline (not used -- crashed on MacBook)
├── features/                     # Cached feature matrices (.npz)
├── checkpoints/                  # Cached 411-dim features for re-runs
├── models/                       # Saved trained models (.joblib)
├── results/
│   ├── metrics/                  # JSON files with all evaluation metrics
│   └── plots/                    # All generated plots (32 PNG files)
├── report/                       # LaTeX report
├── requirements.txt              # Python dependencies
└── EXAM_GUIDE.md                 # This file
```

---

## 13. THE PIPELINE (How It All Flows)

```
Step 1: LOAD        1414 WAV files --> numpy arrays (56,000 numbers each)
                            |
Step 2: EXTRACT     411 features from each clip (MFCC, CQCC, pitch, contrast, chroma, spectral)
                            |
Step 3: SPLIT       80% train (1131 clips) / 20% test (283 clips), stratified
                            |
Step 4: TRAIN       5 models on training data (GMM, SVM, RF, XGBoost, Ensemble)
                            |
Step 5: PREDICT     Each model predicts labels for the 283 test samples
                            |
Step 6: EVALUATE    Accuracy, F1, MCC, Kappa, AUC-ROC, confusion matrices
                            |
Step 7: SAVE        Plots (PNG), metrics (JSON), models (.joblib)
```

**Memory-safe approach (run_ultrasafe.py):**
- Processes **one file at a time** instead of loading all into memory
- Runs `gc.collect()` (garbage collection) after each file
- Limits CPU threads (`OMP_NUM_THREADS=1`) to prevent memory explosion
- Skips pyin pitch tracking (the biggest memory consumer) and estimates F0 from spectral centroid instead
- Caches features in `checkpoints/features_411.npz` so re-runs skip extraction

---

## 14. COMMON VIVA QUESTIONS & ANSWERS

### About the Problem

**Q: What problem are you solving?**
A: Automatic classification of infant cry audio into 5 categories (hunger, belly_pain, burping, discomfort, tiredness) using classical machine learning on acoustic features.

**Q: Why is this important?**
A: New parents struggle to interpret cries. In NICUs, nurses monitor many babies simultaneously. Automated cry analysis enables faster response, reducing infant stress and caregiver burnout.

**Q: Can this work in real life?**
A: With more data and deployment optimization, yes. Potential applications: NICU monitoring systems, baby monitor apps, parent assistance tools. The 96.5% accuracy on 5 classes is promising.

### About the Data

**Q: Why 8kHz sample rate?**
A: The Donate-a-Cry dataset is natively 8kHz. Baby cries have most acoustic energy below 4kHz. By the **Nyquist theorem**, 8kHz sampling captures all frequencies up to 4kHz. No benefit from upsampling.

**Q: Why 7 seconds?**
A: Standard cry bout duration. Padding/truncating all clips to 7 seconds ensures every clip produces the **same-size feature vector** (411 dimensions), which is required by all ML models.

**Q: How do you handle the class imbalance?**
A: Three strategies: (1) **Data augmentation** (17 techniques) to increase minority classes, (2) **SMOTE** in SVM to create synthetic samples, (3) **class_weight='balanced'** in all models to penalize misclassifying rare classes more.

**Q: What is SMOTE?**
A: Synthetic Minority Over-sampling Technique. It creates new synthetic samples by **interpolating between existing minority class examples**. Like drawing a line between two burping samples in 411-dimensional space and creating a new point along that line.

**Q: How do you prevent data leakage with augmentation?**
A: Augmented files are created from the raw audio BEFORE the train/test split. The split is stratified. In SVM, SMOTE is applied INSIDE each cross-validation fold to prevent information from test folds leaking into training.

### About Features

**Q: Why not just use raw audio?**
A: 56,000 raw numbers per clip is too much for classical ML. Most of that data is noise. Feature extraction condenses the audio into 411 **meaningful** numbers that describe its acoustic character. Classical ML cannot learn from raw waveforms -- that requires deep learning (Phase 2).

**Q: What is the Mel scale?**
A: A **frequency scale that matches human pitch perception**. Low frequencies are spread out (we can hear fine differences between 100 Hz and 200 Hz), high frequencies are compressed (we cannot easily tell 5000 Hz from 5100 Hz). MFCCs use this scale.

**Q: What is FFT?**
A: **Fast Fourier Transform**. Converts a time-domain signal (amplitude over time) to frequency-domain (which frequencies are present and how strong they are). It is the mathematical backbone of all spectral features.

**Q: Why use CQCC in addition to MFCC?**
A: CQCC uses Constant-Q Transform which has **variable resolution** -- better frequency resolution at low frequencies (captures F0 and harmonics of baby cries more precisely) and better time resolution at high frequencies (captures transient events like cry onset). This is our key innovation.

**Q: How many total features and why?**
A: 411 features = 240 MFCC + 120 CQCC + 7 pitch + 14 spectral contrast + 24 chroma + 6 spectral descriptors. This comprehensive set captures frequency content (MFCC/CQCC), pitch (F0), harmonic structure (contrast/chroma), and overall sound character (spectral descriptors).

### About Models

**Q: Why not use deep learning?**
A: Phase 1 is classical ML only per the project structure. Phase 2 will use CNN, LSTM, attention mechanisms, and Wav2Vec 2.0 transfer learning. Phase 3 will fuse classical + deep approaches.

**Q: Why is GMM so much worse?**
A: Density estimation in 411 dimensions is extremely hard (**curse of dimensionality**). GMM tries to model the FULL data distribution of each class. With only 1131 training samples in 411 dimensions, there are not enough data points. **Discriminative models** (SVM, RF, XGBoost) only need to find the boundary between classes, not model the full distribution. That is a much easier task.

**Q: What is the difference between Random Forest and XGBoost?**
A: **Random Forest:** builds trees **independently** in parallel, then votes. **XGBoost:** builds trees **sequentially**, each correcting the previous one's errors. Sequential error correction makes XGBoost more accurate but also more prone to overfitting (mitigated by regularization).

**Q: Why does Ensemble have lower accuracy than XGBoost?**
A: XGBoost is already excellent (96.5%). The ensemble averages it with SVM (92.6%) and RF (92.9%), which **dilutes** the best model's performance. However, Ensemble has the **highest AUC-ROC (0.994)**, meaning its probability estimates are the most reliable.

**Q: What is regularization in XGBoost?**
A: Penalties that prevent overly complex trees to avoid **memorizing** training data. It includes: max_depth (limits tree depth), learning_rate (each tree makes only a small correction), subsample and colsample_bytree (each tree sees only a subset). Like adding a "simplicity bonus" -- the model prefers simpler explanations.

**Q: What is the Dirichlet process in GMM?**
A: An automatic way to determine how many Gaussian components each class needs. Starts with 8, then **prunes unnecessary ones** by setting their weights close to zero. This avoids manually choosing the number of components.

**Q: Why One-vs-One for SVM instead of One-vs-All?**
A: For 5 classes, OvO trains 10 binary classifiers (each pair). Each classifier sees a **balanced binary problem** which is better for SVM. OvAll would train 5 classifiers where each separates one class from all others -- creating imbalanced binary problems.

### About Metrics

**Q: What is the difference between Macro F1 and Weighted F1?**
A: **Macro** = average F1 treating all 5 classes equally (each class contributes 20%). **Weighted** = average weighted by number of test samples per class. Macro **penalizes** poor performance on small classes more. When they are close (like our XGBoost: 0.965 vs 0.965), performance is balanced across classes.

**Q: Why is AUC-ROC important for this application?**
A: In medical applications like NICU monitoring, you want **calibrated probability estimates**. A nurse needs to know the model's confidence: "95% chance of pain" triggers immediate response, while "55% chance of hunger" might warrant waiting. AUC-ROC measures how well the model ranks its confidence levels.

### About Future Work

**Q: What happens in Phase 2?**
A: Deep learning approaches: **CNN** on mel spectrograms (treating audio as images), **LSTM/GRU** for temporal sequence modeling, **attention mechanisms** for focusing on important parts of the cry, and **Wav2Vec 2.0** transfer learning from pre-trained speech models.

**Q: What is Phase 3?**
A: Hybrid fusion of classical + deep models, **deployed on mobile/edge devices** (like a phone app or a smart baby monitor) for real-time inference.

**Q: What are the main libraries used?**
A: **librosa** (audio loading and feature extraction), **scikit-learn** (all ML models, evaluation, preprocessing), **xgboost** (gradient boosting), **imbalanced-learn** (SMOTE), **hmmlearn** (HMM), **numpy** (numerical computation), **matplotlib/seaborn** (plotting), **scipy** (DCT for CQCC, signal filtering).

**Q: What is your train/test split?**
A: **80% train (1131 samples) / 20% test (283 samples)**, stratified to preserve class proportions. Random seed = 42 for reproducibility. The split is saved so all models are evaluated on the exact same test set.

**Q: What would you improve if you had more time?**
A: (1) Collect more real data to reduce dependence on augmentation, (2) Cross-validation instead of single split, (3) Hyperparameter tuning with grid search for all models (currently only SVM has it), (4) Feature selection to reduce dimensionality, (5) More augmentation variety for burping class (only 8 originals, hardest to augment meaningfully).

---

## 15. ONE-LINE CHEAT SHEET (Quick Reference)

| Term | One-line definition |
|------|-------------------|
| **MFCC** | How the sound sounds to human ears (frequency content on Mel scale) |
| **CQCC** | Better MFCC for baby cries (variable time-frequency resolution) |
| **F0** | The fundamental pitch of the cry (Hz) |
| **FFT** | Breaks a sound wave into its component frequencies |
| **Mel Scale** | Frequency scale that matches human pitch perception |
| **CQT** | Like FFT but with variable window sizes (better for harmonic sounds) |
| **Delta/Delta-delta** | Rate of change / acceleration of change of features over time |
| **SMOTE** | Makes synthetic examples of rare classes by interpolation |
| **Augmentation** | Creating modified copies of audio to increase training data |
| **GMM** | Models each class as a cloud of bell curves in feature space |
| **SVM** | Draws the best possible boundary between classes |
| **RF** | 200 trees vote on the answer (ensemble of independent trees) |
| **XGBoost** | Trees that learn from each other's mistakes (sequential boosting) |
| **Ensemble** | Combines 3 models' probability outputs via a meta-learner |
| **Stacking** | Type of ensemble where base model outputs feed a second-level model |
| **Confusion Matrix** | Grid showing correct vs incorrect predictions per class |
| **t-SNE** | Squishes 411 features to 2D for visualization |
| **Precision** | Of what the model predicted as X, how many actually were X |
| **Recall** | Of all actual X samples, how many did the model catch |
| **F1 Score** | Balance between precision and recall (harmonic mean) |
| **MCC** | Correlation between predictions and truth (-1 to +1) |
| **Kappa** | How much better than random guessing (0 to 1) |
| **AUC-ROC** | How well the model ranks its confidence (0.5 to 1.0) |
| **Nyquist** | Max frequency capturable = sample_rate / 2 (8kHz -> 4kHz max) |
| **Curse of Dimensionality** | More features = need exponentially more data for density estimation |
| **Stratified Split** | Train/test split that preserves class proportions in both sets |
| **StandardScaler** | Normalizes features to mean=0, std=1 before feeding to models |
| **RBF Kernel** | Allows SVM to draw curved (non-linear) decision boundaries |
| **Dirichlet Process** | Automatically determines the number of GMM components needed |
| **OvO** | One-vs-One: trains a binary classifier for every pair of classes |
| **Hyperphonation** | Very high-pitched cry mode (> 1000 Hz), indicates pain |
| **PAG** | Brain region (Periaqueductal Gray) that controls crying |

---

**You are now ready. Go ace that presentation.**
