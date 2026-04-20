"""Knowledge distillation and model quantization for edge deployment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

from src.phase2.config import KD_TEMPERATURE, KD_ALPHA, MODELS_DIR, WEIGHT_DECAY


class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation.

    Loss = alpha * T^2 * KL(teacher_soft, student_soft) + (1-alpha) * CE(y, student_hard)
    """

    def __init__(self, temperature=KD_TEMPERATURE, alpha=KD_ALPHA):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, targets):
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)

        kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)
        ce_loss = F.cross_entropy(student_logits, targets)

        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss


def train_with_distillation(teacher, student, train_loader, val_loader,
                             device, epochs=100, lr=1e-3):
    """Train student model using teacher's soft targets."""
    teacher.eval()
    student.train()
    teacher.to(device)
    student.to(device)

    criterion = DistillationLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6,
    )

    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        student.train()
        total_loss = 0

        for mel, domain, labels in train_loader:
            mel = mel.to(device)
            domain = domain.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(mel, domain)

            student_logits = student(mel, domain)
            loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * mel.size(0)

        # Validate
        student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, domain, labels in val_loader:
                logits = student(mel.to(device), domain.to(device))
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())

        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  KD Epoch {epoch:3d}/{epochs} | "
                  f"Loss={total_loss/len(all_labels):.4f} | Val F1={val_f1:.3f}")

    if best_state:
        student.load_state_dict(best_state)
    print(f"  Distillation complete — best val F1: {best_f1:.4f}")
    return student


def quantize_model(model, save_path=None):
    """Post-training dynamic quantization (CPU inference)."""
    model.cpu()
    model.eval()

    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.LSTM}, dtype=torch.qint8,
    )

    if save_path is None:
        save_path = Path(MODELS_DIR) / "student_quantized.pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized.state_dict(), save_path)

    # Size comparison
    orig_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    print(f"  Original model size:   {orig_size / 1024:.1f} KB")
    print(f"  Quantized model saved: {save_path}")

    return quantized


def benchmark_inference(model, sample_mel, sample_domain, device, n_runs=100):
    """Measure inference latency."""
    model.eval()
    model.to(device)
    sample_mel = sample_mel.to(device)
    sample_domain = sample_domain.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(sample_mel, sample_domain)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(sample_mel, sample_domain)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    print(f"  Inference latency: {np.mean(times):.2f} +/- {np.std(times):.2f} ms "
          f"(median: {np.median(times):.2f} ms)")
    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times)),
            "median_ms": float(np.median(times))}
