# Infant State Recognition System -- Video Presentation Script

**Presenters:** Meghavi Rao (230044) and Belal Raza (230094)
**Course:** Deep Learning & Advanced Machine Learning -- Semester 6
**Phase:** 1 (Classical Machine Learning)
**Duration:** ~10 minutes

---

## [SLIDE 1] -- Title Slide

**Meghavi:**

Hello everyone. I am Meghavi Rao, roll number 230044, and with me is Belal Raza, roll number 230094. Today we are presenting Phase 1 of our project: the Infant State Recognition System. This project uses audio analysis and classical machine learning to classify infant cries into five categories -- hunger, belly pain, burping, discomfort, and tiredness. Let us walk you through what we built, how it works, and what we found.

---

## [SLIDE 2] -- Motivation and Problem Statement

**Belal:**

So why does this problem matter? New parents, especially first-time parents, often struggle to understand what their baby needs. A crying infant cannot tell you whether they are hungry or in pain -- and yet, research going back to the 1960s tells us that these cries are acoustically distinct.

In 1968, Wasz-Hockert and colleagues were the first to study infant cry acoustics systematically. They showed that cries carry measurable acoustic signatures tied to the baby's underlying state. More recently, the Dunstan Baby Language hypothesis proposed that infants produce five reflexive sound patterns, each mapping to a specific need -- the "neh" sound for hunger, "eairh" for lower gas, "heh" for discomfort, and so on.

From a neuroscience perspective, infant crying is controlled by the Periaqueductal Gray, or PAG, a structure in the brainstem. The PAG produces three distinct phonation modes: modal voice, which is normal everyday crying; hyperphonation, where the fundamental frequency shoots above 1 kilohertz, typically associated with pain; and dysphonation, which is harsh and turbulent sounding. These modes leave different fingerprints in the audio signal, and that is exactly what our system aims to capture.

---

## [SLIDE 3] -- Project Overview and Pipeline

**Meghavi:**

Here is the high-level pipeline for Phase 1. We start with raw audio data, apply augmentation to balance the classes, extract a rich 411-dimensional feature vector from each clip, and then train and evaluate five different classical ML models. The goal in this phase is to establish strong baselines using traditional approaches, before moving to deep learning in Phases 2 and 3.

---

## [SLIDE 4] -- Dataset

**Belal:**

Our data comes from the Donate-a-Cry corpus, which is a publicly available collection of labeled infant cry recordings. The original dataset had around 457 audio files across our five target classes, but the distribution was heavily imbalanced. Hunger had the most samples, while burping had very few.

To address this, we applied a targeted augmentation strategy. We set a minimum target of 300 samples per class. Any class below that threshold was augmented; the majority class, hunger, which already had 382 samples, was left untouched to avoid worsening the imbalance.

Our augmentation pipeline uses 17 different techniques. These include pitch shifting at plus or minus half, one, and two semitones; time stretching at rates between 0.85x and 1.15x; additive white noise at around 25 to 30 decibels SNR; random gain variation within plus or minus 6 dB; room reverb simulation using a synthetic exponentially decaying impulse response; band-pass filtering to simulate microphone variation; time shifting; and a combination transform that applies pitch shift and noise together.

After augmentation, our final dataset has 1,414 audio clips: 382 hunger, 288 belly pain, 144 burping, 300 discomfort, and 300 tiredness. We split this 80/20 into 1,131 training and 283 test samples using stratified sampling to preserve class proportions.

---

## [SLIDE 5] -- Audio Preprocessing

**Meghavi:**

Before feature extraction, every audio clip goes through the same preprocessing. We resample to 8 kilohertz, which matches the native rate of the dataset and gives us a Nyquist frequency of 4 kilohertz -- more than enough for infant cry analysis since the important spectral content sits well below that. Each clip is fixed to exactly 7 seconds. Shorter clips are zero-padded, and longer ones are truncated. This ensures every sample produces a feature vector of exactly the same dimensionality.

---

## [SLIDE 6] -- Feature Engineering (411-Dimensional Vector)

**Belal:**

Feature engineering is the heart of Phase 1, and this is where we put a lot of our effort. Each audio clip is represented by a 411-dimensional feature vector, composed of six distinct blocks.

The first and largest block is MFCCs -- Mel-Frequency Cepstral Coefficients. We extract 40 MFCCs per frame, compute first and second-order deltas to capture temporal dynamics, and then summarize each with mean and standard deviation across all frames. That gives us 40 coefficients times 2 statistics times 3 types -- MFCC, delta, and delta-delta -- for a total of 240 dimensions.

The second block is our CQCC features -- Constant-Q Cepstral Coefficients. This is a key innovation in our approach. While MFCCs use the Mel scale, which has fixed time-frequency resolution, CQCCs are based on the Constant-Q Transform. The CQT provides higher frequency resolution at low frequencies and higher time resolution at high frequencies. This variable resolution is better suited for capturing the harmonic structure of infant cries, because cry harmonics are densely packed at lower frequencies where we need that extra resolution. We extract 20 CQCC coefficients with deltas and delta-deltas, summarized the same way as MFCCs, giving us 120 dimensions.

---

## [SLIDE 7] -- Feature Engineering (continued)

**Meghavi:**

The remaining blocks round out our feature set. The F0 or pitch contour block gives us 7 dimensions: the mean, standard deviation, maximum, minimum, and range of the fundamental frequency, plus the voiced fraction and the pitch slope. These are critical because, as we mentioned, pain cries tend to have much higher pitch while tiredness cries are lower and more monotone.

Spectral contrast provides 14 dimensions -- the peak-to-valley difference across frequency sub-bands. This tells us how "peaky" versus "flat" the spectrum is at different frequency ranges.

Chroma features give us 24 dimensions, capturing the pitch class distribution -- essentially how energy is spread across the 12 semitones of the musical octave.

Finally, we have 6 spectral descriptors: zero-crossing rate, spectral centroid, spectral rolloff, spectral bandwidth, RMS energy, and spectral flatness. Together, these paint a complete picture of each cry's timbral character.

So in total: 240 plus 120 plus 7 plus 14 plus 24 plus 6 equals 411 dimensions per clip.

---

## [SLIDE 8] -- Model 1: Gaussian Mixture Model

**Belal:**

Let us walk through our models. The first is a Gaussian Mixture Model using a Bayesian approach. We train one GMM per class on that class's feature vectors. At test time, we score a sample against every class model and predict the class with the highest log-likelihood.

We use scikit-learn's BayesianGaussianMixture with a Dirichlet process prior. This is important because it automatically prunes unnecessary components during training -- we set up to 8 components, but the Dirichlet process lets the model figure out how many it actually needs. We also incorporate class-weighted log-priors to handle residual class imbalance.

The GMM achieved 62.2% accuracy and a macro F1 of 0.547. This is our weakest model, and we expected that. GMMs make strong assumptions about feature distributions -- specifically that the data within each class follows a mixture of Gaussians. With a 411-dimensional feature space, those assumptions are hard to satisfy. Still, the AUC-ROC of 0.801 tells us the model captures useful density structure even if its hard predictions are not always correct.

---

## [SLIDE 9] -- Model 2: Support Vector Machine

**Meghavi:**

Our second model is an SVM with an RBF kernel. We built this with a two-layer strategy for handling class imbalance. Layer one is SMOTE -- Synthetic Minority Over-sampling Technique -- which generates synthetic samples for minority classes. Layer two is the balanced class weight parameter in the SVM itself, which adjusts the misclassification penalty based on class frequency. We use a One-vs-One decision function, which trains a separate binary classifier for every pair of classes. For 5 classes, that means 10 binary SVMs voting on each prediction.

We ran 5-fold grid search over the regularization parameter C, the kernel coefficient gamma, and kernel type. Importantly, SMOTE is applied inside each cross-validation fold to prevent data leakage.

The SVM reached 92.6% accuracy with a macro F1 of 0.921 and an AUC-ROC of 0.989. That is a massive jump from the GMM and shows that a discriminative model with proper imbalance handling can do very well on this task.

---

## [SLIDE 10] -- Models 3 and 4: Random Forest and XGBoost

**Belal:**

Next, we trained a Random Forest with 500 trees and an XGBoost classifier with 300 boosting rounds. Random Forest hit 92.9% accuracy -- essentially tied with the SVM -- while XGBoost pushed the accuracy to 96.5%, making it our best individual model.

XGBoost's macro F1 of 0.965 and its MCC of 0.955 are both excellent. The per-class F1 scores are remarkably balanced: 0.96 for hunger, 0.98 for belly pain, 0.97 for burping, 0.96 for discomfort, and 0.97 for tiredness. No single class is being neglected, which tells us the model genuinely learned to distinguish all five cry types.

Both tree-based models also give us feature importance rankings, which we used to understand which acoustic properties matter most for classification.

---

## [SLIDE 11] -- Model 5: Stacking Ensemble

**Meghavi:**

Finally, we built a stacking ensemble that combines the SVM, Random Forest, and XGBoost. The idea is simple: each base model produces a probability vector over the 5 classes for each test sample. We concatenate these probability outputs into a 15-dimensional meta-feature vector and feed it into a Logistic Regression meta-learner that makes the final prediction.

The ensemble achieved 95.1% accuracy with a macro F1 of 0.944 and the highest AUC-ROC in our study at 0.994. Interestingly, the ensemble did not beat XGBoost on accuracy -- it actually scores slightly lower. This can happen when one base model is already very strong and the meta-learner averages in the weaker opinions of the other base models. However, the ensemble's AUC-ROC of 0.994 is the best we have, meaning it produces the most well-calibrated probability estimates overall.

---

## [SLIDE 12] -- Results Summary Table

**Belal:**

Here is the full comparison table. Let me highlight the key numbers.

The GMM sits at 62.2% accuracy -- a useful baseline but clearly limited. The SVM and Random Forest are nearly tied around 92 to 93%. XGBoost is the clear winner on hard classification metrics at 96.5% accuracy, 0.965 macro F1, and 0.955 MCC. The stacking ensemble leads on AUC-ROC at 0.994 but trades a small amount of accuracy for that.

We used six metrics deliberately. Accuracy alone can be misleading with imbalanced classes, which is why we also report macro F1, which weights all classes equally; weighted F1, which accounts for class sizes; Matthew's Correlation Coefficient, which is robust to imbalance; Cohen's Kappa, which adjusts for chance agreement; and AUC-ROC, which measures ranking quality.

---

## [SLIDE 13] -- Analysis and Visualizations

**Meghavi:**

Our t-SNE visualization of the 411-dimensional feature space shows clear clustering by cry type, which visually confirms that our features carry strong discriminative information. There is some overlap between hunger and discomfort, which makes clinical sense -- a hungry baby often sounds uncomfortable.

The feature importance plots from both Random Forest and XGBoost show that MFCC features dominate the top rankings, but CQCC features and pitch statistics also appear prominently. This validates our decision to go beyond standard MFCCs and include the Constant-Q-based features and F0 contour.

The confusion matrices for XGBoost show very few off-diagonal errors, with the largest confusion occurring between hunger and discomfort -- consistent with what we see in the t-SNE plot and consistent with the clinical overlap between these two states.

---

## [SLIDE 14] -- Key Takeaways from Phase 1

**Belal:**

Let us summarize our main takeaways from Phase 1.

First, feature engineering matters enormously. The jump from a 13-MFCC baseline to our 411-dimensional vector, with CQCCs, pitch, contrast, chroma, and spectral descriptors, is what made 96.5% accuracy possible.

Second, data augmentation was essential. Going from around 457 files with severe imbalance to 1,414 files with a balanced distribution made a real difference in model stability.

Third, discriminative models drastically outperform generative ones on this task. The GMM's density estimation approach struggled in 411 dimensions, while SVM, Random Forest, and XGBoost all exceeded 92%.

Fourth, XGBoost is our best single model, but the stacking ensemble offers the best calibrated probabilities, which would matter more in a clinical deployment where you want reliable confidence scores.

---

## [SLIDE 15] -- Future Work: Phase 2 and Phase 3

**Meghavi:**

Looking ahead, Phase 2 will move into deep learning. We plan to train convolutional neural networks on Mel spectrogram images and LSTM networks on the temporal MFCC sequences. We also want to experiment with attention mechanisms that can focus on the most informative segments of each cry.

In Phase 3, we aim to fine-tune a pre-trained Wav2Vec 2.0 model on our dataset. Wav2Vec learns powerful audio representations from large-scale self-supervised pre-training, and fine-tuning it on infant cries could push our accuracy even further. The ultimate goal is a real-time mobile application where a parent can hold up their phone, record their baby's cry, and get an immediate prediction of what the baby needs.

---

## [SLIDE 16] -- Closing

**Belal:**

To wrap up: in Phase 1, we built a complete classical ML pipeline for infant cry classification. We went from raw audio to a 411-dimensional feature vector, trained five models, and achieved 96.5% accuracy with XGBoost. Our CQCC features, augmentation strategy, and multi-metric evaluation give us confidence that these results are robust and not artifacts of overfitting or class imbalance.

Thank you for watching. We welcome any questions or feedback.

---

*End of script. Total approximate word count: ~2,200 words. Estimated speaking time at natural pace: 10 minutes.*
