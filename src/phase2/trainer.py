"""Training loop with warmup, LDAM+DRW, early stopping, and decoupled training."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score

from src.phase2.config import (
    LR, LR_MIN, WARMUP_EPOCHS, WARMUP_LR, LR_PATIENCE, LR_FACTOR,
    EARLY_STOP_PATIENCE, WEIGHT_DECAY, LABEL_SMOOTHING,
    DRW_SWITCH_EPOCH, MAX_EPOCHS_STUDENT, BATCH_SIZE,
    CLASS_COUNTS, NUM_CLASSES, MIXUP_ALPHA,
)
from src.phase2.losses import LDAMLoss, DRWScheduler


def mixup_batch(mel, domain, labels, alpha=MIXUP_ALPHA):
    """Within-batch mixup."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(mel.size(0))
    mel_mix = lam * mel + (1 - lam) * mel[idx]
    dom_mix = lam * domain + (1 - lam) * domain[idx]
    return mel_mix, dom_mix, labels, labels[idx], lam


class Trainer:
    """Handles the full training loop including warmup, DRW, and early stopping."""

    def __init__(self, model, device, cls_num_list=None,
                 use_ldam=True, max_epochs=MAX_EPOCHS_STUDENT,
                 lr=LR, label_smoothing=LABEL_SMOOTHING):
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.lr = lr

        if cls_num_list is None:
            cls_num_list = CLASS_COUNTS

        if use_ldam:
            self.criterion = LDAMLoss(
                cls_num_list, max_m=0.5, s=30.0,
                label_smoothing=label_smoothing,
            ).to(device)
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
            ).to(device)

        self.use_ldam = use_ldam
        self.drw = DRWScheduler(cls_num_list, switch_epoch=DRW_SWITCH_EPOCH)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=LR_FACTOR,
            patience=LR_PATIENCE, min_lr=LR_MIN,
        )
        self.history = defaultdict(list)
        self.best_val_f1 = 0.0
        self.best_state = None
        self.patience_counter = 0

    def _warmup_lr(self, epoch):
        if epoch < WARMUP_EPOCHS:
            alpha = epoch / WARMUP_EPOCHS
            lr = WARMUP_LR + alpha * (self.lr - WARMUP_LR)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self._warmup_lr(epoch)

        total_loss = 0
        all_preds = []
        all_labels = []

        for mel, domain, labels in train_loader:
            mel = mel.to(self.device)
            domain = domain.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            use_mixup = np.random.random() < 0.5
            if use_mixup:
                mel, domain, labels_a, labels_b, lam = mixup_batch(mel, domain, labels)

            logits = self.model(mel, domain)

            if self.use_ldam:
                drw_w = self.drw.get_weights(epoch).to(self.device)
                if use_mixup:
                    loss_a = self.criterion(logits, labels_a, reduction='none')
                    loss_b = self.criterion(logits, labels_b, reduction='none')
                    w_a = drw_w[labels_a]
                    w_b = drw_w[labels_b]
                    loss = (lam * (loss_a * w_a) + (1 - lam) * (loss_b * w_b)).mean()
                else:
                    per_sample_loss = self.criterion(logits, labels, reduction='none')
                    per_sample_w = drw_w[labels]
                    loss = (per_sample_loss * per_sample_w).mean()
            else:
                if use_mixup:
                    loss = (lam * F.cross_entropy(logits, labels_a)
                            + (1 - lam) * F.cross_entropy(logits, labels_b))
                else:
                    loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * mel.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            target_labels = labels_a if use_mixup else labels
            all_labels.extend(target_labels.cpu().numpy())

        avg_loss = total_loss / len(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, macro_f1

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for mel, domain, labels in val_loader:
            mel = mel.to(self.device)
            domain = domain.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(mel, domain)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * mel.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, macro_f1

    def fit(self, train_loader, val_loader, verbose=True):
        """Full training loop with early stopping."""
        for epoch in range(self.max_epochs):
            t0 = time.time()
            train_loss, train_f1 = self.train_epoch(train_loader, epoch)
            val_loss, val_f1 = self.validate(val_loader)

            self.scheduler.step(val_f1)
            cur_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_f1"].append(train_f1)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(cur_lr)

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
                marker = " *"
            else:
                self.patience_counter += 1
                marker = ""

            if verbose and (epoch % 5 == 0 or marker):
                drw_status = "DRW" if epoch >= DRW_SWITCH_EPOCH else "UNI"
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:3d}/{self.max_epochs} | "
                      f"Train L={train_loss:.4f} F1={train_f1:.3f} | "
                      f"Val L={val_loss:.4f} F1={val_f1:.3f} | "
                      f"LR={cur_lr:.1e} [{drw_status}] "
                      f"({elapsed:.1f}s){marker}")

            if self.patience_counter >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best val F1: {self.best_val_f1:.4f})")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    def decoupled_retrain(self, train_loader, val_loader, crt_epochs=30, verbose=True):
        """Freeze backbone, retrain only fusion + classifier with class-balanced sampling."""
        if verbose:
            print("\n  === Decoupled Training (cRT) ===")

        # Freeze backbone, keep only fusion/classifier layers trainable
        for name, param in self.model.named_parameters():
            if any(k in name for k in ["fusion", "binary_head", "fine_head", "classifier"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if verbose:
            print(f"  Trainable params (fusion only): {trainable}")

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr * 0.1, weight_decay=WEIGHT_DECAY,
        )

        for epoch in range(crt_epochs):
            train_loss, train_f1 = self.train_epoch(train_loader, epoch + self.max_epochs)
            val_loss, val_f1 = self.validate(val_loader)

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                marker = " *"
            else:
                marker = ""

            if verbose and (epoch % 5 == 0 or marker):
                print(f"  cRT Epoch {epoch:3d}/{crt_epochs} | "
                      f"Val F1={val_f1:.3f}{marker}")

        # Unfreeze all
        for param in self.model.parameters():
            param.requires_grad = True

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self.history


class HierarchicalTrainer:
    """Train the two-stage hierarchical model."""

    def __init__(self, model, device, cls_num_list=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        )
        self.binary_criterion = nn.CrossEntropyLoss()
        self.fine_criterion = LDAMLoss(
            [16, 8, 27, 24], max_m=0.5, s=30.0,
        ).to(device) if cls_num_list is None else LDAMLoss(
            cls_num_list, max_m=0.5, s=30.0,
        ).to(device)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        n = 0

        for mel, domain, labels in train_loader:
            mel = mel.to(self.device)
            domain = domain.to(self.device)
            labels = labels.to(self.device)

            binary_labels = (labels > 0).long()  # 0=hunger, 1=non-hunger
            binary_logits, fine_logits = self.model(mel, domain)

            loss_binary = self.binary_criterion(binary_logits, binary_labels)

            non_hunger_mask = labels > 0
            if non_hunger_mask.any():
                fine_labels = labels[non_hunger_mask] - 1  # map 1-4 to 0-3
                fine_out = fine_logits[non_hunger_mask]
                loss_fine = self.fine_criterion(fine_out, fine_labels)
                loss = loss_binary + loss_fine
            else:
                loss = loss_binary

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * mel.size(0)
            n += mel.size(0)

        return total_loss / n

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        for mel, domain, labels in val_loader:
            mel = mel.to(self.device)
            domain = domain.to(self.device)
            preds = self.model.predict(mel, domain).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return macro_f1
