"""Interpretability: Grad-CAM, attention visualization, and feature importance."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.phase2.config import CLASSES, PLOTS_DIR, MEL_N_MELS


# ── Grad-CAM ────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN layers."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, mel_spec, domain_feats, target_class=None):
        """Generate Grad-CAM heatmap. Returns (heatmap, predicted_class)."""
        self.model.eval()
        mel_spec.requires_grad_(True)

        logits = self.model(mel_spec, domain_feats)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=mel_spec.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, target_class


def plot_gradcam(mel_spec_np, cam, true_label, pred_label, save_path=None):
    """Overlay Grad-CAM heatmap on mel-spectrogram."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(mel_spec_np, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Mel-Spectrogram")
    axes[0].set_ylabel("Mel Bin")

    axes[1].imshow(cam, aspect="auto", origin="lower", cmap="jet")
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(mel_spec_np, aspect="auto", origin="lower", cmap="magma")
    axes[2].imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay (True={CLASSES[true_label]}, Pred={CLASSES[pred_label]})")

    for ax in axes:
        ax.set_xlabel("Time Frame")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def generate_gradcam_gallery(model, dataloader, device, save_dir=None, n_per_class=5):
    """Generate Grad-CAM visualizations for n samples per class."""
    if save_dir is None:
        save_dir = Path(PLOTS_DIR) / "gradcam"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find the last CNN layer
    cnn_layers = list(model.cnn.children())
    target_layer = cnn_layers[-1].conv.pointwise  # last depthwise sep conv

    grad_cam = GradCAM(model, target_layer)

    class_counts = {i: 0 for i in range(len(CLASSES))}
    model.eval()

    for mel, domain, labels in dataloader:
        for i in range(mel.size(0)):
            label = labels[i].item()
            if class_counts[label] >= n_per_class:
                continue

            mel_i = mel[i:i+1].to(device)
            dom_i = domain[i:i+1].to(device)

            cam, pred = grad_cam.generate(mel_i, dom_i)
            mel_np = mel[i, 0].numpy()

            path = save_dir / f"gradcam_{CLASSES[label]}_{class_counts[label]}.png"
            plot_gradcam(mel_np, cam, label, pred, save_path=path)
            class_counts[label] += 1

        if all(c >= n_per_class for c in class_counts.values()):
            break

    print(f"  Grad-CAM gallery saved to {save_dir}/")


# ── Attention weight visualization ───────────────────────────────────

def plot_attention_weights(model, dataloader, device, save_dir=None, n_per_class=5):
    """Visualize temporal attention weights overlaid on spectrograms."""
    if save_dir is None:
        save_dir = Path(PLOTS_DIR) / "attention"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    class_weights = {i: [] for i in range(len(CLASSES))}

    with torch.no_grad():
        for mel, domain, labels in dataloader:
            mel_dev = mel.to(device)
            weights = model.get_attention_weights(mel_dev).cpu().numpy()

            for i in range(mel.size(0)):
                label = labels[i].item()
                if len(class_weights[label]) < n_per_class:
                    class_weights[label].append((mel[i, 0].numpy(), weights[i]))

    # Per-class attention plots
    for cls_idx, samples in class_weights.items():
        if not samples:
            continue
        fig, axes = plt.subplots(len(samples), 1, figsize=(12, 3 * len(samples)))
        if len(samples) == 1:
            axes = [axes]
        for j, (mel_np, w) in enumerate(samples):
            axes[j].imshow(mel_np, aspect="auto", origin="lower", cmap="magma")
            ax2 = axes[j].twinx()
            ax2.plot(np.linspace(0, mel_np.shape[1], len(w)), w, color="cyan", alpha=0.8, lw=2)
            ax2.set_ylabel("Attention")
            axes[j].set_ylabel("Mel Bin")
        fig.suptitle(f"Attention Weights — {CLASSES[cls_idx]}")
        plt.tight_layout()
        path = save_dir / f"attention_{CLASSES[cls_idx]}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

    # Aggregate attention per class
    fig, ax = plt.subplots(figsize=(10, 4))
    for cls_idx, samples in class_weights.items():
        if not samples:
            continue
        all_w = np.array([w for _, w in samples])
        mean_w = all_w.mean(axis=0)
        ax.plot(mean_w, label=CLASSES[cls_idx], lw=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Average Attention Pattern by Class")
    ax.legend()
    plt.tight_layout()
    path = save_dir / "attention_aggregate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"  Attention plots saved to {save_dir}/")


# ── Domain feature importance (simple permutation-based) ─────────────

def domain_feature_importance(model, dataloader, device, feature_names=None):
    """Permutation importance for the 32-dim domain features."""
    model.eval()

    # Baseline score
    all_preds, all_labels = [], []
    all_mels, all_doms = [], []
    with torch.no_grad():
        for mel, domain, labels in dataloader:
            all_mels.append(mel)
            all_doms.append(domain)
            all_labels.extend(labels.numpy())
            logits = model(mel.to(device), domain.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())

    from sklearn.metrics import f1_score
    baseline_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    all_mels = torch.cat(all_mels, dim=0)
    all_doms = torch.cat(all_doms, dim=0)
    all_labels = np.array(all_labels)

    n_features = all_doms.shape[1]
    importances = []

    for feat_idx in range(n_features):
        perm_doms = all_doms.clone()
        perm_doms[:, feat_idx] = perm_doms[torch.randperm(len(perm_doms)), feat_idx]

        with torch.no_grad():
            logits = model(all_mels.to(device), perm_doms.to(device))
            perm_preds = logits.argmax(1).cpu().numpy()

        perm_f1 = f1_score(all_labels, perm_preds, average="macro", zero_division=0)
        importances.append(baseline_f1 - perm_f1)

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    sorted_idx = np.argsort(importances)[::-1]
    result = [(feature_names[i], importances[i]) for i in sorted_idx]
    return result
