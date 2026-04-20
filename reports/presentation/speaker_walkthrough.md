# Speaker Walkthrough — Infant State Recognition System

Read this before recording. Each section maps to a slide. Speak naturally, not word-for-word.

---

## Slide 1 — Title

"Hi, this is our project on Infant State Recognition. We built a system that classifies baby cries into five categories. The dataset has 457 recordings with a massive 48-to-1 class imbalance. Our team is Meghavi Rao and Belal Raza."

---

## Slide 2 — The Problem

"Babies cry differently depending on what they need. There are five types based on Dunstan Baby Language: hunger, belly pain, burping, discomfort, and tiredness. Each one comes from a different reflex in the baby's body. The challenge is that our dataset is extremely skewed — 382 out of 457 recordings are hunger cries. Burping has just 8 samples. That's a 48-to-1 ratio."

---

## Slide 3 — Dataset Breakdown

"Here's the exact breakdown. Hunger is 83.6% of the data. The four minority classes together make up less than 17%. We have 221 unique infants, each clip is 7 seconds at 8kHz. For Phase 2 we split by infant ID — no baby appears in more than one split. This gives us a 70/15/15 train/val/test ratio."

---

## Slide 4 — Data Leakage Discovery

"Early on, our XGBoost scored 96.5% accuracy. That was suspicious. We found the problem: our pipeline was augmenting audio files before splitting, so the model was tested on near-copies of its training data. Some infants had up to 13 recordings scattered across train and test. Once we fixed this with GroupShuffleSplit by infant ID, XGBoost dropped from 96.5% to 17.8% macro-F1. The honest best result was SVM at 27% macro-F1."

---

## Slide 5 — Phase 1 Features

"For Phase 1, we built a 411-dimensional feature vector. It has six blocks: MFCCs capture the spectral shape of the cry. CQCCs give better frequency resolution where baby cry harmonics live. F0 contour tells us about pitch dynamics — pain cries have high-pitched sounds above 1000 Hz. Spectral contrast measures harmonic richness. Chroma and spectral descriptors round it out."

---

## Slide 6 — Phase 1 Results

"We trained six classical ML models. SVM with SMOTE was the best at 27% macro-F1. That sounds low, but look at the confusion matrix — it predicts hunger for almost everything. Only discomfort gets partially recognised with F1 of 0.44. The other three minority classes all score zero. The Ensemble gets the highest accuracy at 83.7%, but that's misleading — a model that always predicts hunger would get 84%."

---

## Slide 7 — The Accuracy Trap

"This is the key insight. 81.5% accuracy sounds decent, but macro-F1 is only 27%. Why? Because 3 out of 4 minority classes have F1 of zero. The model basically learned to say 'hunger' for everything. That's why we use macro-F1 as our primary metric — it treats all five classes equally."

---

## Slide 8 — Phase 2 Architecture

"For Phase 2, we built a CNN + BiLSTM + Temporal Attention model. The CNN processes mel-spectrograms — that's a visual representation of the audio showing frequency over time. BiLSTM captures temporal patterns in both directions. Temporal attention learns which parts of the cry are most important for classification. We also have a separate branch for 32 domain features like pitch, voice quality, and MFCCs."

---

## Slide 9 — Imbalance Techniques

"We used six techniques together to fight the imbalance. LDAM loss pushes decision boundaries away from minority classes. Deferred Re-Weighting uses uniform weights first, then switches to class-balanced weights at epoch 60. We also used Mixup, SpecAugment, data augmentation to balance classes at 384 each, and decoupled classifier retraining."

---

## Slide 10 — DL Results

"Here's the surprising result: deep learning alone could not beat the simple SVM. Only the Teacher model with 135K parameters barely passed SVM at 29.3% macro-F1. All the student-scale models at 16K parameters performed worse. The training curves show why — massive overfitting. Training F1 reaches 65% but validation stays at 18%."

---

## Slide 11 — Why DL Failed

"Three reasons. First, 326 training samples is far too few for CNN+LSTM architectures. Augmentation creates variants of the same 8 burping templates — it can't create genuinely new patterns. Second, the validation set has only 63 samples, mostly hunger, so early stopping signals are pure noise. Third, despite six regularisation techniques, overfitting was severe."

---

## Slide 12 — The Hybrid Breakthrough

"This is where it gets exciting. We combined SVM probabilities with DL probabilities through a weighted average. SVM gets 35% weight, DL gets 30%. The result: macro-F1 jumps from 27% to 50.7% — that's an 87.5% relative improvement. Accuracy hits 92.6%. This works because SVM and DL make different mistakes on different samples."

---

## Slide 13 — Hybrid Confusion Matrices

"Look at these confusion matrices. The weighted ensemble gets all 61 hunger samples correct, plus it recognises some discomfort and tiredness samples — classes that were completely invisible before. The stacking variant does even better on minority classes but misclassifies some hunger samples."

---

## Slide 14 — Why Hybrid Works

"SVM captures global spectral patterns reliably — it's great at hunger. DL on spectrograms occasionally catches temporal patterns that handcrafted features miss. When you fuse their probability outputs, high-confidence predictions from each model survive while errors get averaged out. The spec-only ablation variant got discomfort F1 of 0.40, proving the spectrogram branch learns something the 411-dim features cannot."

---

## Slide 15 — Ablation Study

"We ran 11 ablation variants. Key findings: removing BiLSTM drops performance by 20%, confirming temporal patterns matter. Removing attention drops it by 13%. LDAM beats standard cross-entropy. Without augmentation, the model gets highest accuracy but lowest macro-F1 — it just predicts hunger even more. And interestingly, domain features actually hurt DL performance."

---

## Slide 16 — Interpretability

"The attention patterns show the model learns meaningful things. Burping cries get attention peaks in the middle of the cry, matching the short explosive nature of the Eh reflex. Tiredness focuses on the early part, matching the yawn-like Owh onset. For feature importance, higher-order MFCCs dominate, while F0 statistics barely matter in the DL context."

---

## Slide 17 — Edge Deployment

"We distilled the Teacher into a 16K-parameter Student and quantised it to INT8. The final model is 35.9 KB — that fits on an ESP32 microcontroller. Inference takes 11.4 milliseconds, which is 600 times faster than real-time. This makes it viable for baby monitors, phones, and embedded devices."

---

## Slide 18 — Final Comparison

"Here's the mandatory three-model comparison. Model A, the SVM baseline: macro-F1 of 27%. Model B, DL alone: 18.2% — actually worse. Model C, the hybrid: 50.7%, nearly doubling the baseline. The hybrid wins on every single metric."

---

## Slide 19 — Key Takeaways

"Three lessons. One: fix your data first. GroupShuffleSplit dropped us from 96% to 27%, but gave honest results. Two: deep learning is not magic. With 457 samples and 48:1 imbalance, SVM beats every DL model we tried. Three: combine strengths. The hybrid ensemble worked because diverse signals beat deeper models."

---

## Slide 20 — Limitations

"We're honest about limitations. The test set is tiny — 68 samples with zero belly pain. Results are on one corpus only. The Random Forest component was broken. And we didn't try transfer learning from large pretrained audio models, which could help significantly."

---

## Slide 21 — Thank You

"To summarise: we achieved macro-F1 of 0.507, an 87.5% improvement, with a model that's only 35.9 KB. The key insight is that hybrid ML+DL ensembles significantly outperform either approach alone when data is scarce. Thank you."
