"""Two-stage decoupled retraining (cRT, Kang et al. ICLR 2020) applied to
the PAD_UFES20 cross_attention headline model - VAL-ONLY, does not touch
the test split.

Stage 1 (representation learning) reuses the existing locked headline
checkpoints (logs/PAD_UFES20/checkpoints/cross_attention_seed{seed}_best.pt)
unchanged - they already ARE "the full model trained exactly as the
current headline configuration" (train_cross_attention_fusion.py: same
warm-start, same LR/batch/epoch/loss settings). Each is copied to
cross_attention_crt_stage1_seed{seed}_best.pt so the locked headline file
is never opened for writing. (Retraining Stage 1 from scratch would only
re-derive the same procedure on the same data at large CPU cost with no
methodological difference - flagged to the user rather than done silently;
available on request.)

Stage 2 (classifier re-training) freezes every parameter except
model.head (FC(320->128) -> BatchNorm -> ReLU -> Dropout -> FC(128->6))
and retrains only the head on the TRAINING set using class-balanced
sampling (WeightedRandomSampler with inverse-class-frequency sample
weights, so each class is equally likely per batch - replaces Stage 1's
class-weighted loss, per Kang et al.: balanced sampling substitutes for
the loss re-weighting once only the head is being fit). Up to 15 epochs,
early stopping (patience 3) on validation macro-F1.

Usage:
    python -m src.models.train_cross_attention_crt --seed 0
    python -m src.models.train_cross_attention_crt --seed 0 --seed 1 --seed 2
"""

import argparse
import csv
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.config import BATCH_SIZE, WEIGHT_DECAY, get_dataset
from src.models.cross_attention_fusion_model import CrossAttentionFusionModel
from src.models.dataset import FusionDataset, MetadataPreprocessor
from src.models.train import set_seed

STAGE2_MAX_EPOCHS = 15
STAGE2_EARLY_STOPPING_PATIENCE = 3
STAGE2_LR = 1e-4  # head-only; higher than the 1e-5 full-model fine-tune LR
                   # since only ~41k params are being fit over a short schedule


def build_balanced_sampler(train_ds: FusionDataset, ds_config) -> WeightedRandomSampler:
    labels = train_ds.df["disease_label"].map(ds_config.label_to_idx).to_numpy()
    class_counts = np.bincount(labels, minlength=ds_config.num_classes)
    class_weight = 1.0 / class_counts
    sample_weights = class_weight[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, metadata, labels in loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_labels


def evaluate_val(model, val_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, metadata, labels in val_loader:
            outputs = model(images.to(device), metadata.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    return macro_f1, dict(zip(class_names, per_class_f1.tolist()))


def run_seed(seed: int) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset("PAD_UFES20")
    class_names = ds_config.class_names

    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
    train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)
    val_ds = FusionDataset(ds_config.val_csv, ds_config, preprocessor, train=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- Stage 1: reuse the locked headline checkpoint (never opened for
    # writing), copied under a new name.
    locked_ckpt = ds_config.checkpoints_dir / f"cross_attention_seed{seed}_best.pt"
    stage1_ckpt = ds_config.checkpoints_dir / f"cross_attention_crt_stage1_seed{seed}_best.pt"
    if not locked_ckpt.exists():
        raise FileNotFoundError(f"Locked headline checkpoint not found: {locked_ckpt}")
    if not stage1_ckpt.exists():
        shutil.copy2(locked_ckpt, stage1_ckpt)

    model = CrossAttentionFusionModel(
        metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
    )
    checkpoint = torch.load(stage1_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    stage1_val_macro_f1, stage1_val_per_class_f1 = evaluate_val(model, val_loader, device, class_names)

    # --- Stage 2: freeze everything except the classifier head.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[seed {seed}] stage 2: {n_trainable} trainable params (head only)")

    sampler = build_balanced_sampler(train_ds, ds_config)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

    criterion = torch.nn.CrossEntropyLoss()  # unweighted - balanced sampling replaces the reweighting
    optimizer = torch.optim.Adam(trainable, lr=STAGE2_LR, weight_decay=WEIGHT_DECAY)

    stage2_ckpt = ds_config.checkpoints_dir / f"cross_attention_crt_stage2_seed{seed}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_cross_attention_crt_stage2_seed{seed}.csv"

    best_val_macro_f1 = -1.0
    best_val_per_class_f1 = None
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_macro_f1", "val_macro_f1"])

        for epoch in range(1, STAGE2_MAX_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1, _, _ = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_macro_f1, val_per_class_f1 = evaluate_val(model, val_loader, device, class_names)
            writer.writerow([epoch, train_loss, train_macro_f1, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[seed {seed}] stage2 epoch {epoch:02d} train_loss={train_loss:.4f} "
                f"train_macroF1={train_macro_f1:.4f} val_macroF1={val_macro_f1:.4f} ({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_val_per_class_f1 = val_per_class_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": "PAD_UFES20",
                        "branch": "cross_attention_crt_stage2",
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": ds_config.num_classes,
                        "metadata_input_dim": preprocessor.output_dim,
                        "stage1_checkpoint": str(stage1_ckpt),
                    },
                    stage2_ckpt,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= STAGE2_EARLY_STOPPING_PATIENCE:
                    print(f"[seed {seed}] stage2 early stopping at epoch {epoch}")
                    break

    result = {
        "seed": seed,
        "stage1_val_macro_f1": stage1_val_macro_f1,
        "stage1_val_per_class_f1": stage1_val_per_class_f1,
        "stage2_val_macro_f1": best_val_macro_f1,
        "stage2_val_per_class_f1": best_val_per_class_f1,
        "stage1_checkpoint": str(stage1_ckpt),
        "stage2_checkpoint": str(stage2_ckpt),
    }
    summary_path = ds_config.logs_dir / f"train_cross_attention_crt_stage2_seed{seed}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[seed {seed}] stage1 val macro-F1={stage1_val_macro_f1:.4f} -> "
        f"stage2 val macro-F1={best_val_macro_f1:.4f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="cRT-style two-stage decoupled retraining (PAD_UFES20 cross_attention)")
    parser.add_argument("--seed", type=int, action="append", required=True, dest="seeds")
    args = parser.parse_args()

    results = [run_seed(seed) for seed in args.seeds]

    out_dir = get_dataset("PAD_UFES20").fusion_reports_dir.parent / "cross_attention_crt"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "crt_val_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"combined results -> {out_dir / 'crt_val_results.json'}")


if __name__ == "__main__":
    main()
