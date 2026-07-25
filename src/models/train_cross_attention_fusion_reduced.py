"""Phase 8 reduced-feature cross-attention fusion training - PAD-UFES-20 only.

Usage:
    python -m src.models.train_cross_attention_fusion_reduced --dataset PAD_UFES20 --seed 0

Schema-matched counterpart to train_cross_attention_fusion.py for the
PAD-UFES-20 -> HAM10000 cross-dataset generalization experiment - same
reduced-feature/site-normalization rationale as train_fusion_reduced.py.
Warm-starts the image embedder from the existing, unchanged
image_seed{N}_best.pt and the metadata embedder from the new
metadata_reduced_seed{N}_best.pt.

Saved as cross_attention_reduced_seed{N}_best.pt, alongside (not
overwriting) Phase 7 Stage 2's cross_attention_seed{N}_best.pt, which
remains the checkpoint used for every already-reported PAD-UFES-20-
internal cross-attention result.
"""

import argparse
import csv
import json
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_CROSS_ATTENTION,
    NUM_EPOCHS,
    REDUCED_CATEGORICAL_FEATURES,
    REDUCED_NUMERIC_FEATURES,
    WEIGHT_DECAY,
    get_dataset,
    normalize_anatomical_site_for_cross_dataset,
)
from src.models.cross_attention_fusion_model import CrossAttentionFusionModel
from src.models.dataset import FusionDataset, MetadataPreprocessor
from src.models.train import compute_class_weights, set_seed
from src.models.train_cross_attention_fusion import run_epoch_cross_attention


def train_one_run(dataset_name: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)
    reduced_config = ds_config.with_features(REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES)

    train_df = pd.read_csv(reduced_config.train_csv)
    preprocessor = MetadataPreprocessor(
        reduced_config,
        column_transforms={"anatomical_site": normalize_anatomical_site_for_cross_dataset},
    ).fit(train_df)

    train_ds = FusionDataset(reduced_config.train_csv, reduced_config, preprocessor, train=True)
    val_ds = FusionDataset(reduced_config.val_csv, reduced_config, preprocessor, train=False)

    model = CrossAttentionFusionModel(
        metadata_input_dim=preprocessor.output_dim, num_classes=reduced_config.num_classes
    ).to(device)

    image_checkpoint = ds_config.stage1_checkpoints_dir / f"image_seed{seed}_best.pt"
    metadata_checkpoint = ds_config.checkpoints_dir / f"metadata_reduced_seed{seed}_best.pt"
    for path in (image_checkpoint, metadata_checkpoint):
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path} - reduced-feature cross-attention "
                f"warm-start requires the existing Stage 1 image checkpoint and "
                f"the new metadata_reduced checkpoint (train_metadata_reduced.py) "
                f"to exist first."
            )
    model.load_stage1_checkpoints(image_checkpoint, metadata_checkpoint, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        reduced_config.train_csv, reduced_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_CROSS_ATTENTION, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"cross_attention_reduced_seed{seed}"
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
            train_loss, train_macro_f1 = run_epoch_cross_attention(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch_cross_attention(
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
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": "cross_attention_reduced",
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": reduced_config.num_classes,
                        "metadata_input_dim": preprocessor.output_dim,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": "cross_attention_reduced",
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "warm_start_image_checkpoint": str(image_checkpoint),
        "warm_start_metadata_checkpoint": str(metadata_checkpoint),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 reduced-feature cross-attention fusion training"
    )
    parser.add_argument("--dataset", choices=["PAD_UFES20"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
