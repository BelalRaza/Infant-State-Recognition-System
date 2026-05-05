"""Safely adapt AST to infant cry/non-cry audio before 5-class training.

This stage uses only rows approved for auxiliary pretraining. It learns a
cry-domain checkpoint without treating weak/unlabeled audio as 5-class labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2a.config import AST_MODEL_NAME, AUX_RESULTS_DIR, HF_CACHE_DIR, ROOT_DIR, UNIFIED_MANIFEST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=UNIFIED_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=AUX_RESULTS_DIR)
    parser.add_argument("--project-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--model-name", default=AST_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--unfreeze-last-n", type=int, default=0, help="0=head-only safest; 1-2=true AST domain adaptation.")
    parser.add_argument("--include-external-train", action="store_true", help="Also use non-eval external cause rows as cry examples.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from src.phase2a.auxiliary import AuxiliaryAudioDataset, load_auxiliary_manifest, make_ast_collate_fn, split_auxiliary_manifest
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    args.output_dir.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_auxiliary_manifest(args.manifest, include_external_train=args.include_external_train)
    manifest = split_auxiliary_manifest(manifest)
    manifest.to_csv(args.output_dir / "auxiliary_binary_manifest.csv", index=False)
    print("Auxiliary rows:")
    print(manifest.groupby(["aux_split", "binary_label"]).size().unstack(fill_value=0))

    processor = ASTFeatureExtractor.from_pretrained(args.model_name, cache_dir=HF_CACHE_DIR)
    model = ASTForAudioClassification.from_pretrained(
        args.model_name,
        cache_dir=HF_CACHE_DIR,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(device)
    configure_trainable_layers(model, args.unfreeze_last_n)

    train_df = manifest[manifest["aux_split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["aux_split"] == "val"].reset_index(drop=True)
    collate_fn = make_ast_collate_fn(processor)
    train_loader = DataLoader(
        AuxiliaryAudioDataset(train_df, args.project_root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        AuxiliaryAudioDataset(val_df, args.project_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(make_optimizer_groups(model, args.lr_head, args.lr_backbone), weight_decay=args.weight_decay)
    start_epoch = 1
    best_f1 = -1.0
    last_ckpt = args.output_dir / "last_checkpoint.pt"
    if last_ckpt.exists() and not args.force:
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        best_f1 = float(state.get("best_f1", -1.0))
        print(f"Resuming from epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(row)
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_f1": max(best_f1, val_metrics["macro_f1"]),
                "args": vars(args),
            },
            last_ckpt,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_dir = args.output_dir / "best_ast_binary_adapter"
            model.audio_spectrogram_transformer.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            (best_dir / "README.md").write_text(
                "AST encoder adapted only on safe auxiliary cry-vs-noncry labels. "
                "Use this as an initialization/feature extractor, not as a 5-class classifier.\n",
                encoding="utf-8",
            )

    (args.output_dir / "auxiliary_adaptation_metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best adapted AST checkpoint: {args.output_dir / 'best_ast_binary_adapter'}")


def configure_trainable_layers(model, unfreeze_last_n: int) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    if unfreeze_last_n <= 0:
        return
    base = model.audio_spectrogram_transformer
    layers = list(base.encoder.layer)
    for layer in layers[-unfreeze_last_n:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in base.layernorm.parameters():
        param.requires_grad = True


def make_optimizer_groups(model, lr_head: float, lr_backbone: float) -> list[dict]:
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return groups


def train_one_epoch(model, loader, optimizer, device) -> float:
    import numpy as np
    import torch
    from tqdm import tqdm

    model.train()
    losses = []
    for batch in tqdm(loader, desc="aux-train"):
        labels = batch.pop("labels").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(**batch, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(out.loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, device) -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    from tqdm import tqdm

    model.eval()
    labels_all = []
    preds_all = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="aux-val"):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=1)
            labels_all.extend(labels.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())
    labels_np = np.asarray(labels_all)
    preds_np = np.asarray(preds_all)
    precision, recall, f1, support = precision_recall_fscore_support(labels_np, preds_np, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "macro_f1": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "noncry_f1": float(f1[0]),
        "cry_f1": float(f1[1]),
        "noncry_support": int(support[0]),
        "cry_support": int(support[1]),
        "noncry_precision": float(precision[0]),
        "cry_precision": float(precision[1]),
        "noncry_recall": float(recall[0]),
        "cry_recall": float(recall[1]),
    }


if __name__ == "__main__":
    main()

