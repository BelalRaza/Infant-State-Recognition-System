"""Model architectures: CNN baseline, CNN+BiLSTM+Attention, Teacher, Hierarchical.

All models follow the parameter budget from the master plan:
  Student: ~40K params
  Teacher: ~200K params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.phase2.config import (
    NUM_CLASSES, DOMAIN_FEAT_DIM, DROPOUT, SE_REDUCTION,
    STUDENT_CNN_CHANNELS, STUDENT_LSTM_HIDDEN, STUDENT_ATTN_DIM,
    STUDENT_FUSION_DIM, STUDENT_FEAT_HIDDEN,
    TEACHER_CNN_CHANNELS, TEACHER_LSTM_HIDDEN, TEACHER_ATTN_DIM,
    TEACHER_FUSION_DIM, TEACHER_FEAT_HIDDEN,
)


# ── Building blocks ─────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=SE_REDUCTION):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DepthSepConv2d(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class CNNBlock(nn.Module):
    """DepthSepConv + LayerNorm + ReLU + SE + MaxPool + Dropout."""

    def __init__(self, in_ch, out_ch, dropout=DROPOUT):
        super().__init__()
        self.conv = DepthSepConv2d(in_ch, out_ch)
        self.norm = nn.GroupNorm(1, out_ch)  # GroupNorm(1) == LayerNorm for conv
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = self.se(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class TemporalAttention(nn.Module):
    """Additive attention over temporal dimension."""

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.W(lstm_out))  # (batch, seq, attn_dim)
        scores = self.v(energy).squeeze(-1)     # (batch, seq)
        weights = F.softmax(scores, dim=1)      # (batch, seq)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        return context, weights


# ── CNN-only baseline (ablation variant 6) ───────────────────────────

class CNNBaseline(nn.Module):
    """Mel-spectrogram-only CNN with global average pooling."""

    def __init__(self, n_classes=NUM_CLASSES, channels=None, dropout=DROPOUT):
        super().__init__()
        if channels is None:
            channels = STUDENT_CNN_CHANNELS
        self.cnn = nn.Sequential(
            CNNBlock(1, channels[0], dropout),
            CNNBlock(channels[0], channels[1], dropout),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], n_classes),
        )

    def forward(self, mel_spec, domain_feats=None):
        x = self.cnn(mel_spec)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)


# ── CNN + BiLSTM + Attention (spectrogram branch only) ───────────────

class CNNBiLSTMAttention(nn.Module):
    """Full spectrogram branch without domain feature fusion."""

    def __init__(self, n_classes=NUM_CLASSES, channels=None,
                 lstm_hidden=STUDENT_LSTM_HIDDEN, attn_dim=STUDENT_ATTN_DIM,
                 dropout=DROPOUT):
        super().__init__()
        if channels is None:
            channels = STUDENT_CNN_CHANNELS
        self.cnn = nn.Sequential(
            CNNBlock(1, channels[0], dropout),
            CNNBlock(channels[0], channels[1], dropout),
        )
        self.lstm_hidden = lstm_hidden
        # After 2x MaxPool(2,2): freq_bins = n_mels/4, time_frames = T/4
        # Reshape: (batch, time_frames/4, channels[-1]*freq_bins/4)
        # Project to lstm input dim
        self.proj = None  # lazy init
        self.lstm = nn.LSTM(
            input_size=lstm_hidden * 2,  # will be set by proj
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        self.attention = TemporalAttention(lstm_hidden * 2, attn_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, n_classes),
        )
        self._proj_inited = False
        self._cnn_channels = channels[-1]

    def _init_proj(self, cnn_out):
        b, c, f, t = cnn_out.shape
        in_dim = c * f
        device = cnn_out.device
        self.proj = nn.Linear(in_dim, self.lstm_hidden * 2).to(device)
        self.lstm.input_size = self.lstm_hidden * 2
        self._proj_inited = True

    def forward(self, mel_spec, domain_feats=None):
        x = self.cnn(mel_spec)  # (B, C, F', T')
        if not self._proj_inited:
            self._init_proj(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F')
        x = self.proj(x)  # (B, T', lstm_hidden*2)
        lstm_out, _ = self.lstm(x)  # (B, T', lstm_hidden*2)
        context, attn_weights = self.attention(lstm_out)
        return self.classifier(context)

    def get_attention_weights(self, mel_spec):
        """Return attention weights for visualization."""
        x = self.cnn(mel_spec)
        if not self._proj_inited:
            self._init_proj(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        _, weights = self.attention(lstm_out)
        return weights


# ── Multi-Input Fusion Model (full architecture) ────────────────────

class MultiInputCryModel(nn.Module):
    """CNN+BiLSTM+Attention on mel-spec FUSED with domain feature branch."""

    def __init__(self, n_classes=NUM_CLASSES, channels=None,
                 lstm_hidden=STUDENT_LSTM_HIDDEN, attn_dim=STUDENT_ATTN_DIM,
                 fusion_dim=STUDENT_FUSION_DIM, feat_hidden=STUDENT_FEAT_HIDDEN,
                 domain_dim=DOMAIN_FEAT_DIM, dropout=DROPOUT):
        super().__init__()
        if channels is None:
            channels = STUDENT_CNN_CHANNELS
        self.cnn = nn.Sequential(
            CNNBlock(1, channels[0], dropout),
            CNNBlock(channels[0], channels[1], dropout),
        )
        self.lstm_hidden = lstm_hidden
        self.proj = None
        self.lstm = nn.LSTM(
            input_size=lstm_hidden * 2,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = TemporalAttention(lstm_hidden * 2, attn_dim)

        # Domain feature branch
        self.feat_branch = nn.Sequential(
            nn.Linear(domain_dim, feat_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feat_hidden),
        )

        # Fusion
        spec_dim = lstm_hidden * 2
        self.fusion = nn.Sequential(
            nn.Linear(spec_dim + feat_hidden, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_classes),
        )
        self._proj_inited = False

    def _init_proj(self, cnn_out):
        b, c, f, t = cnn_out.shape
        device = cnn_out.device
        self.proj = nn.Linear(c * f, self.lstm_hidden * 2).to(device)
        self._proj_inited = True

    def _spec_forward(self, mel_spec):
        x = self.cnn(mel_spec)
        if not self._proj_inited:
            self._init_proj(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        return context, attn_weights

    def forward(self, mel_spec, domain_feats):
        spec_vec, _ = self._spec_forward(mel_spec)
        feat_vec = self.feat_branch(domain_feats)
        fused = torch.cat([spec_vec, feat_vec], dim=1)
        return self.fusion(fused)

    def get_attention_weights(self, mel_spec):
        x = self.cnn(mel_spec)
        if not self._proj_inited:
            self._init_proj(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        _, weights = self.attention(lstm_out)
        return weights

    def get_embeddings(self, mel_spec, domain_feats):
        """Return (spec_embedding, feat_embedding) before fusion."""
        spec_vec, _ = self._spec_forward(mel_spec)
        feat_vec = self.feat_branch(domain_feats)
        return spec_vec, feat_vec


# ── Teacher Model (wider, ~200K params) ──────────────────────────────

def build_teacher(n_classes=NUM_CLASSES):
    return MultiInputCryModel(
        n_classes=n_classes,
        channels=TEACHER_CNN_CHANNELS,
        lstm_hidden=TEACHER_LSTM_HIDDEN,
        attn_dim=TEACHER_ATTN_DIM,
        fusion_dim=TEACHER_FUSION_DIM,
        feat_hidden=TEACHER_FEAT_HIDDEN,
    )


def build_student(n_classes=NUM_CLASSES):
    return MultiInputCryModel(
        n_classes=n_classes,
        channels=STUDENT_CNN_CHANNELS,
        lstm_hidden=STUDENT_LSTM_HIDDEN,
        attn_dim=STUDENT_ATTN_DIM,
        fusion_dim=STUDENT_FUSION_DIM,
        feat_hidden=STUDENT_FEAT_HIDDEN,
    )


# ── Feature-only MLP (ablation variant 5) ───────────────────────────

class FeatureOnlyMLP(nn.Module):
    """MLP on 32-dim domain features only (no spectrogram branch)."""

    def __init__(self, n_classes=NUM_CLASSES, domain_dim=DOMAIN_FEAT_DIM,
                 hidden=64, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(domain_dim, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, mel_spec, domain_feats):
        return self.net(domain_feats)


# ── Spectrogram-only (no fusion, ablation variant 4) ────────────────

class SpecOnlyModel(nn.Module):
    """CNN+BiLSTM+Attention on spectrogram, no domain features."""

    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = CNNBiLSTMAttention(**kwargs)

    def forward(self, mel_spec, domain_feats=None):
        return self.backbone(mel_spec)


# ── No-Attention variant (ablation 2) ────────────────────────────────

class NoAttentionModel(nn.Module):
    """CNN+BiLSTM with last hidden state, no attention."""

    def __init__(self, n_classes=NUM_CLASSES, channels=None,
                 lstm_hidden=STUDENT_LSTM_HIDDEN,
                 feat_hidden=STUDENT_FEAT_HIDDEN,
                 fusion_dim=STUDENT_FUSION_DIM,
                 domain_dim=DOMAIN_FEAT_DIM, dropout=DROPOUT):
        super().__init__()
        if channels is None:
            channels = STUDENT_CNN_CHANNELS
        self.cnn = nn.Sequential(
            CNNBlock(1, channels[0], dropout),
            CNNBlock(channels[0], channels[1], dropout),
        )
        self.lstm_hidden = lstm_hidden
        self.proj = None
        self.lstm = nn.LSTM(
            input_size=lstm_hidden * 2,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.feat_branch = nn.Sequential(
            nn.Linear(domain_dim, feat_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feat_hidden),
        )
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + feat_hidden, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_classes),
        )
        self._proj_inited = False

    def _init_proj(self, cnn_out):
        b, c, f, t = cnn_out.shape
        device = cnn_out.device
        self.proj = nn.Linear(c * f, self.lstm_hidden * 2).to(device)
        self._proj_inited = True

    def forward(self, mel_spec, domain_feats):
        x = self.cnn(mel_spec)
        if not self._proj_inited:
            self._init_proj(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.proj(x)
        _, (h_n, _) = self.lstm(x)
        spec_vec = torch.cat([h_n[0], h_n[1]], dim=1)
        feat_vec = self.feat_branch(domain_feats)
        fused = torch.cat([spec_vec, feat_vec], dim=1)
        return self.fusion(fused)


# ── No-BiLSTM variant (ablation 3) ──────────────────────────────────

class NoBiLSTMModel(nn.Module):
    """CNN with GAP + domain features, no LSTM."""

    def __init__(self, n_classes=NUM_CLASSES, channels=None,
                 feat_hidden=STUDENT_FEAT_HIDDEN,
                 fusion_dim=STUDENT_FUSION_DIM,
                 domain_dim=DOMAIN_FEAT_DIM, dropout=DROPOUT):
        super().__init__()
        if channels is None:
            channels = STUDENT_CNN_CHANNELS
        self.cnn = nn.Sequential(
            CNNBlock(1, channels[0], dropout),
            CNNBlock(channels[0], channels[1], dropout),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_branch = nn.Sequential(
            nn.Linear(domain_dim, feat_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feat_hidden),
        )
        self.fusion = nn.Sequential(
            nn.Linear(channels[-1] + feat_hidden, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_classes),
        )

    def forward(self, mel_spec, domain_feats):
        x = self.cnn(mel_spec)
        x = self.gap(x).view(x.size(0), -1)
        feat_vec = self.feat_branch(domain_feats)
        fused = torch.cat([x, feat_vec], dim=1)
        return self.fusion(fused)


# ── Hierarchical two-stage model ─────────────────────────────────────

class HierarchicalModel(nn.Module):
    """Two-stage: binary (hunger vs non-hunger) -> 4-class fine-grained."""

    def __init__(self, backbone, n_fine_classes=4, threshold=0.3):
        super().__init__()
        self.backbone = backbone
        self.threshold = threshold

        # Find the fusion input dim by inspecting the backbone
        fusion_layers = list(backbone.fusion.children())
        fusion_in = fusion_layers[0].in_features

        self.binary_head = nn.Linear(fusion_in, 2)
        self.fine_head = nn.Linear(fusion_in, n_fine_classes)

    def _get_fusion_input(self, mel_spec, domain_feats):
        spec_vec, _ = self.backbone._spec_forward(mel_spec)
        feat_vec = self.backbone.feat_branch(domain_feats)
        return torch.cat([spec_vec, feat_vec], dim=1)

    def forward(self, mel_spec, domain_feats):
        fused = self._get_fusion_input(mel_spec, domain_feats)
        binary_logits = self.binary_head(fused)
        fine_logits = self.fine_head(fused)
        return binary_logits, fine_logits

    def predict(self, mel_spec, domain_feats):
        """Hierarchical prediction: returns 5-class labels."""
        binary_logits, fine_logits = self.forward(mel_spec, domain_feats)
        binary_probs = F.softmax(binary_logits, dim=1)
        p_non_hunger = binary_probs[:, 1]

        fine_preds = torch.argmax(fine_logits, dim=1)
        # Map fine classes: 0=belly_pain, 1=burping, 2=discomfort, 3=tiredness
        # In original 5-class: hunger=0, belly_pain=1, burping=2, discomfort=3, tiredness=4
        fine_to_full = torch.tensor([1, 2, 3, 4], device=fine_preds.device)
        mapped_fine = fine_to_full[fine_preds]

        is_non_hunger = p_non_hunger > self.threshold
        preds = torch.where(is_non_hunger, mapped_fine, torch.zeros_like(mapped_fine))
        return preds


# ── Utilities ────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(variant="full_model", n_classes=NUM_CLASSES, **kwargs):
    """Factory function for building model variants."""
    if variant == "full_model":
        return build_student(n_classes)
    elif variant == "no_attention":
        return NoAttentionModel(n_classes=n_classes)
    elif variant == "no_bilstm":
        return NoBiLSTMModel(n_classes=n_classes)
    elif variant == "spec_only":
        return SpecOnlyModel(n_classes=n_classes)
    elif variant == "feat_only":
        return FeatureOnlyMLP(n_classes=n_classes)
    elif variant == "cnn_only":
        return CNNBaseline(n_classes=n_classes)
    elif variant == "teacher_200k":
        return build_teacher(n_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")
