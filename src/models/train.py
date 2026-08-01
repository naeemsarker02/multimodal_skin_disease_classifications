"""Phase 6 Stage 1 training entrypoint - dataset-parameterized baselines.

Usage:
    python -m src.models.train --dataset PAD_UFES20 --branch image --seed 0
    python -m src.models.train --dataset HAM10000 --branch metadata --seed 0

Trains exactly one dataset/branch/seed combination per invocation. Uses
the train split for gradient updates and the val split for model
selection (early stopping, checkpoint picking). The test split is never
loaded here - it is read only by src/evaluation/evaluate.py, in a
separate, later, final run. See Project_Tracking.md decision (4) for the
full val/test discipline rationale.

Every dataset uses the identical recipe (architecture, hyperparameters,
class-weighted loss mechanism, seeds) - only the dataset's own paths,
class list, and metadata feature list differ (src/models/config.py).
This is intentional, per Project_Tracking.md's "Sequencing Decision -
HAM10000 Baseline Before Phase 7" entry: keeping the recipe fixed across
datasets means any later cross-dataset performance difference (Phase 8)
can be attributed to the dataset itself, not to per-dataset tuning.
"""

import argparse
import csv
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.backbones import BACKBONE_NAMES, build_backbone
from src.models.config import (
    BATCH_SIZE,
    DATASETS,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_IMAGE,
    LEARNING_RATE_METADATA,
    NUM_EPOCHS,
    STRONG_AUGMENT_TARGET_CLASSES,
    WEIGHT_DECAY,
    get_dataset,
)
from src.models.dataset import ImageDataset, MetadataDataset, MetadataPreprocessor
from src.models.metadata_model import MetadataMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_csv, class_names) -> torch.Tensor:
    """Inverse class frequency, computed from the train split only."""
    df = pd.read_csv(train_csv)
    counts = df["disease_label"].value_counts()
    freqs = np.array([counts.get(name, 0) for name in class_names], dtype=np.float64)
    weights = freqs.sum() / (len(class_names) * freqs)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def _image_run_name(seed: int, backbone: str, sampler: str, strong_augment: str) -> str:
    """image_seed{N} for the pre-Phase-8B/Step-3a default (backbone=
    efficientnet_b0, sampler=shuffle, strong_augment=none), so every
    existing checkpoint/log filename (and Phase 7 fusion's warm-start path)
    keeps working unchanged. Non-default choices are appended so Phase 8B's
    5-backbone runs and Step 3a's imbalance-ablation runs never collide on
    disk.
    """
    parts = ["image"]
    if backbone != "efficientnet_b0":
        parts.append(backbone)
    if sampler != "shuffle":
        parts.append(sampler)
    if strong_augment != "none":
        parts.append(strong_augment)
    parts.append(f"seed{seed}")
    return "_".join(parts)


def train_one_run(
    dataset_name: str,
    branch: str,
    seed: int,
    backbone: str = "efficientnet_b0",
    sampler: str = "shuffle",
    strong_augment: str = "none",
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    if ds_config.image_branch_only and branch != "image":
        raise ValueError(
            f"{dataset_name!r} is image_branch_only=True (its train_csv includes "
            f"rows with no compatible clinical metadata - see "
            f"Project_Tracking.md, 'Step 2 Integration Plan', 2026-07-29). "
            f"--branch {branch!r} would silently train on all-NaN metadata for "
            f"those rows. Use --branch image, or --dataset PAD_UFES20 for "
            f"metadata/fusion training."
        )

    if branch == "image":
        strong_augment_classes = (
            set(STRONG_AUGMENT_TARGET_CLASSES) if strong_augment == "minority" else None
        )
        train_ds = ImageDataset(
            ds_config.train_csv, ds_config, train=True,
            strong_augment_classes=strong_augment_classes,
        )
        val_ds = ImageDataset(ds_config.val_csv, ds_config, train=False)
        model = build_backbone(backbone, num_classes=ds_config.num_classes).to(device)
        lr = LEARNING_RATE_IMAGE
        run_name = _image_run_name(seed, backbone, sampler, strong_augment)
    elif branch == "metadata":
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
        train_ds = MetadataDataset(ds_config.train_csv, ds_config, preprocessor)
        val_ds = MetadataDataset(ds_config.val_csv, ds_config, preprocessor)
        model = MetadataMLP(
            input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        ).to(device)
        lr = LEARNING_RATE_METADATA
        run_name = f"{branch}_seed{seed}"
    else:
        raise ValueError(f"Unknown branch: {branch}")

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)

    if branch == "image" and sampler == "weighted":
        # Step 3a ablation (a): oversample rare classes via inverse
        # train-class-frequency weights (same numbers compute_class_weights
        # already derives for the loss) - reused here as per-sample sampling
        # weights instead. Mutually exclusive with shuffle=True per
        # DataLoader's own constraint.
        train_df = pd.read_csv(ds_config.train_csv)
        sample_weights = train_df["disease_label"].map(
            lambda name: class_weights[ds_config.label_to_idx[name]].item()
        ).to_numpy()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1 = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_macro_f1, val_loss, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[{dataset_name}/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "dataset": dataset_name,
                    "branch": branch,
                    "seed": seed,
                    "epoch": epoch,
                    "val_macro_f1": val_macro_f1,
                    "num_classes": ds_config.num_classes,
                }
                if branch == "image":
                    checkpoint["backbone"] = backbone
                    checkpoint["sampler"] = sampler
                    checkpoint["strong_augment"] = strong_augment
                torch.save(checkpoint, checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": branch,
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
    }
    if branch == "image":
        summary["backbone"] = backbone
        summary["sampler"] = sampler
        summary["strong_augment"] = strong_augment
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Stage 1 / Phase 8B / Step 3a training")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--branch", choices=["image", "metadata"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--backbone", choices=BACKBONE_NAMES, default="efficientnet_b0",
        help="Phase 8B 5-backbone comparison (image branch only). Ignored for --branch metadata.",
    )
    parser.add_argument(
        "--sampler", choices=["shuffle", "weighted"], default="shuffle",
        help="Step 3a imbalance ablation (a): WeightedRandomSampler (image branch only).",
    )
    parser.add_argument(
        "--strong-augment", choices=["none", "minority"], default="none",
        dest="strong_augment",
        help="Step 3a imbalance ablation (b): stronger augmentation for "
             "STRONG_AUGMENT_TARGET_CLASSES only (image branch only).",
    )
    args = parser.parse_args()
    train_one_run(
        args.dataset, args.branch, args.seed,
        backbone=args.backbone, sampler=args.sampler, strong_augment=args.strong_augment,
    )


if __name__ == "__main__":
    main()
