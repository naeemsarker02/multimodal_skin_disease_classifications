"""Supervised Contrastive Loss (SupCon, Khosla et al. NeurIPS 2020) added
to the PAD_UFES20 cross_attention headline training recipe - VAL-ONLY,
does not touch the test split.

L = CE_loss (class-weighted, same weights as the headline) + lambda * SupCon_loss
(lambda=0.1 by default), applied to the 320-d fused (post-cross-attention,
pre-classifier) feature vector (CrossAttentionFusionModel.forward's `joint`
tensor, obtained via return_features=True - see cross_attention_fusion_model.py).

This is a NEW training run from scratch: same architecture, same warm-start
protocol (from the Stage 1 image/metadata checkpoints, identical to
train_cross_attention_fusion.py), same data, same 3 seeds, same LR/batch/
epoch/early-stopping settings - only the loss function changes. Writes to
new checkpoint/log names (cross_attention_supcon_*) so the locked headline
checkpoints (cross_attention_seed{seed}_best.pt) are never opened for
writing.

Usage:
    python -m src.models.train_cross_attention_supcon --seed 0
    python -m src.models.train_cross_attention_supcon --seed 0 --seed 1 --seed 2 --lambda-supcon 0.1
"""

import argparse
import csv
import json
import time

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_CROSS_ATTENTION,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    get_dataset,
)
from src.models.cross_attention_fusion_model import CrossAttentionFusionModel
from src.models.dataset import FusionDataset, MetadataPreprocessor
from src.models.losses import SupConLoss
from src.models.train import compute_class_weights, set_seed

SUPCON_TEMPERATURE = 0.07  # Khosla et al. 2020 default


def run_epoch(model, loader, ce_criterion, supcon_criterion, lambda_supcon, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_ce, total_supcon = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, metadata, labels in loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits, features = model(images, metadata, return_features=True)
            ce_loss = ce_criterion(logits, labels)
            supcon_loss = supcon_criterion(features, labels)
            loss = ce_loss + lambda_supcon * supcon_loss
            if train:
                loss.backward()
                optimizer.step()
            n = labels.size(0)
            total_loss += loss.item() * n
            total_ce += ce_loss.item() * n
            total_supcon += supcon_loss.item() * n
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    n_total = len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n_total, total_ce / n_total, total_supcon / n_total, macro_f1


def evaluate_val(model, val_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, metadata, labels in val_loader:
            logits = model(images.to(device), metadata.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    return macro_f1, dict(zip(class_names, per_class_f1.tolist()))


def train_one_run(seed: int, lambda_supcon: float) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset("PAD_UFES20")
    class_names = ds_config.class_names

    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
    train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)
    val_ds = FusionDataset(ds_config.val_csv, ds_config, preprocessor, train=False)

    model = CrossAttentionFusionModel(
        metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
    ).to(device)

    image_checkpoint = ds_config.stage1_checkpoints_dir / f"image_seed{seed}_best.pt"
    metadata_checkpoint = ds_config.stage1_checkpoints_dir / f"metadata_seed{seed}_best.pt"
    for path in (image_checkpoint, metadata_checkpoint):
        if not path.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {path} - cross-attention "
                f"warm-start requires both branches' Stage 1 checkpoints to exist first."
            )
    model.load_stage1_checkpoints(image_checkpoint, metadata_checkpoint, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(ds_config.train_csv, ds_config.class_names).to(device)
    ce_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    supcon_criterion = SupConLoss(temperature=SUPCON_TEMPERATURE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_CROSS_ATTENTION, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"cross_attention_supcon_seed{seed}"
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    best_val_per_class_f1 = None
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_ce", "train_supcon", "train_macro_f1",
             "val_loss", "val_ce", "val_supcon", "val_macro_f1"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_ce, train_supcon, train_macro_f1 = run_epoch(
                model, train_loader, ce_criterion, supcon_criterion, lambda_supcon, optimizer, device, train=True
            )
            val_loss, val_ce, val_supcon, val_macro_f1 = run_epoch(
                model, val_loader, ce_criterion, supcon_criterion, lambda_supcon, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_ce, train_supcon, train_macro_f1,
                              val_loss, val_ce, val_supcon, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[PAD_UFES20/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} (ce={train_ce:.4f} supcon={train_supcon:.4f}) "
                f"train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} ({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                _, best_val_per_class_f1 = evaluate_val(model, val_loader, device, class_names)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": "PAD_UFES20",
                        "branch": "cross_attention_supcon",
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "lambda_supcon": lambda_supcon,
                        "supcon_temperature": SUPCON_TEMPERATURE,
                        "num_classes": ds_config.num_classes,
                        "metadata_input_dim": preprocessor.output_dim,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[PAD_UFES20/{run_name}] early stopping at epoch {epoch}")
                    break

    result = {
        "seed": seed,
        "lambda_supcon": lambda_supcon,
        "val_macro_f1": best_val_macro_f1,
        "val_per_class_f1": best_val_per_class_f1,
        "checkpoint_path": str(checkpoint_path),
        "warm_start_image_checkpoint": str(image_checkpoint),
        "warm_start_metadata_checkpoint": str(metadata_checkpoint),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[PAD_UFES20/{run_name}] best val macro-F1: {best_val_macro_f1:.4f} -> {checkpoint_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="SupCon-augmented cross-attention fusion training (PAD_UFES20, val-only)")
    parser.add_argument("--seed", type=int, action="append", required=True, dest="seeds")
    parser.add_argument("--lambda-supcon", type=float, default=0.1)
    args = parser.parse_args()

    results = [train_one_run(seed, args.lambda_supcon) for seed in args.seeds]

    out_dir = get_dataset("PAD_UFES20").fusion_reports_dir.parent / "cross_attention_supcon"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "supcon_val_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"combined results -> {out_dir / 'supcon_val_results.json'}")


if __name__ == "__main__":
    main()
