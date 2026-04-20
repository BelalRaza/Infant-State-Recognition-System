"""Hybrid ML+DL ensemble: stacking and weighted average fusion."""

import torch
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.phase2.config import NUM_CLASSES, MODELS_DIR, CLASSES


class HybridEnsemble:
    """Stacking ensemble combining Phase 1 ML models with Phase 2 DL model.

    Level 0: SVM probs (5) + RF probs (5) + DL probs (5) = 15-dim
    Level 1: Logistic Regression meta-classifier
    """

    def __init__(self):
        self.meta_clf = None
        self.ml_models = {}

    def load_ml_models(self, models_dir=None):
        """Load trained Phase 1 models from disk."""
        if models_dir is None:
            models_dir = Path(MODELS_DIR).parent
        models_dir = Path(models_dir)

        model_files = {
            "SVM": "svm_classifier.joblib",
            "RF": "rf_classifier.joblib",
        }
        for name, fname in model_files.items():
            path = models_dir / fname
            if path.exists():
                self.ml_models[name] = joblib.load(path)
                print(f"  Loaded {name} from {path}")
            else:
                print(f"  [WARN] {path} not found, skipping {name}")

    def get_ml_probas(self, X_features_411):
        """Get probability outputs from Phase 1 ML models.
        X_features_411: (N, 411) array of Phase 1 features.
        """
        probas = []
        for name, model in self.ml_models.items():
            try:
                p = model.predict_proba(X_features_411)
                probas.append(p)
            except Exception as e:
                print(f"  [WARN] {name} predict_proba failed: {e}")
                probas.append(np.zeros((len(X_features_411), NUM_CLASSES)))
        return probas

    def get_dl_probas(self, dl_model, dataloader, device):
        """Get probability outputs from DL model."""
        dl_model.eval()
        all_probs = []
        with torch.no_grad():
            for mel, domain, _ in dataloader:
                mel = mel.to(device)
                domain = domain.to(device)
                logits = dl_model(mel, domain)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)

    def fit(self, ml_probas_list, dl_probas, val_labels):
        """Train meta-classifier on validation set predictions.

        ml_probas_list: list of (N, 5) arrays from ML models
        dl_probas: (N, 5) array from DL model
        val_labels: (N,) true labels
        """
        meta_features = np.concatenate(ml_probas_list + [dl_probas], axis=1)
        print(f"  Meta-feature shape: {meta_features.shape}")

        self.meta_clf = LogisticRegression(
            max_iter=1000, multi_class="multinomial",
            class_weight="balanced", random_state=42,
        )
        self.meta_clf.fit(meta_features, val_labels)

        meta_preds = self.meta_clf.predict(meta_features)
        f1 = f1_score(val_labels, meta_preds, average="macro", zero_division=0)
        print(f"  Meta-classifier val MacF1: {f1:.4f}")

    def predict(self, ml_probas_list, dl_probas):
        """Predict using the stacking ensemble."""
        meta_features = np.concatenate(ml_probas_list + [dl_probas], axis=1)
        return self.meta_clf.predict(meta_features)

    def predict_proba(self, ml_probas_list, dl_probas):
        meta_features = np.concatenate(ml_probas_list + [dl_probas], axis=1)
        return self.meta_clf.predict_proba(meta_features)

    def save(self, path=None):
        if path is None:
            path = Path(MODELS_DIR) / "hybrid_ensemble.joblib"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.meta_clf, path)
        print(f"  Hybrid ensemble saved: {path}")


class WeightedEnsemble:
    """Simple weighted average of model probability outputs."""

    def __init__(self):
        self.weights = None

    def fit(self, ml_probas_list, dl_probas, val_labels):
        """Grid-search for optimal weights on validation set."""
        best_f1 = 0
        best_w = None
        n_ml = len(ml_probas_list)

        for w_dl in np.arange(0.3, 0.9, 0.1):
            remaining = 1.0 - w_dl
            for w_split in np.arange(0, remaining + 0.05, remaining / max(n_ml, 1)):
                w_ml = [w_split] + [remaining - w_split] * (n_ml - 1) if n_ml > 1 else [remaining]
                if len(w_ml) != n_ml:
                    continue

                avg_probs = w_dl * dl_probas
                for i, p in enumerate(ml_probas_list):
                    avg_probs = avg_probs + w_ml[i] * p

                preds = avg_probs.argmax(axis=1)
                f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = {"dl": w_dl, "ml": w_ml}

        self.weights = best_w
        print(f"  Weighted ensemble — best weights: DL={best_w['dl']:.2f}, "
              f"ML={[f'{w:.2f}' for w in best_w['ml']]}")
        print(f"  Val MacF1: {best_f1:.4f}")

    def predict(self, ml_probas_list, dl_probas):
        avg_probs = self.weights["dl"] * dl_probas
        for i, p in enumerate(ml_probas_list):
            avg_probs = avg_probs + self.weights["ml"][i] * p
        return avg_probs.argmax(axis=1)

    def predict_proba(self, ml_probas_list, dl_probas):
        avg_probs = self.weights["dl"] * dl_probas
        for i, p in enumerate(ml_probas_list):
            avg_probs = avg_probs + self.weights["ml"][i] * p
        return avg_probs
