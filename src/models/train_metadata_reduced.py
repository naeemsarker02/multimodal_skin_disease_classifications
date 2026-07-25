"""Phase 8 reduced-feature metadata training - PAD-UFES-20 only.

Usage:
    python -m src.models.train_metadata_reduced --dataset PAD_UFES20 --seed 0

Trains a metadata-only model on PAD-UFES-20 restricted to the 3 columns
HAM10000 also has (age, sex, anatomical_site - config.REDUCED_NUMERIC_FEATURES/
REDUCED_CATEGORICAL_FEATURES), with anatomical_site normalized into
HAM10000's vocabulary (config.normalize_anatomical_site_for_cross_dataset,
per docs/Phase8_Anatomical_Site_Mapping.csv, approved 2026-07-18).

Purpose-built for Phase 8's PAD-UFES-20 -> HAM10000 cross-dataset
generalization experiment - the existing rich-feature (21-column)
metadata_seed{N}_best.pt checkpoints cannot run on HAM10000 data at all
(18 of PAD-UFES-20's columns don't exist there), so this schema-matched
variant is a separate, additional model, not a replacement. Saved as
metadata_reduced_seed{N}_best.pt, alongside (not overwriting) the existing
metadata_seed{N}_best.pt used for every already-reported PAD-UFES-20-
internal result.

Mirrors train.py's metadata branch exactly (same architecture,
hyperparameters, class-weighted loss mechanism, seeds) - only the
DatasetConfig's feature lists and the preprocessor's anatomical_site
normalization differ. Same loss/metric/seed/split discipline as every
prior stage: class-weighted CrossEntropyLoss, macro-F1 for model
selection/early stopping, SEEDS = [0, 1, 2], identical train/val split
files. Test split untouched.
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
    LEARNING_RATE_METADATA,
    NUM_EPOCHS,
    REDUCED_CATEGORICAL_FEATURES,
    REDUCED_NUMERIC_FEATURES,
    WEIGHT_DECAY,
    get_dataset,
    normalize_anatomical_site_for_cross_dataset,
)
from src.models.dataset import MetadataDataset, MetadataPreprocessor
from src.models.metadata_model import MetadataMLP
from src.models.train import compute_class_weights, run_epoch, set_seed


def build_reduced_preprocessor(ds_config, train_df: pd.DataFrame):
    reduced_config = ds_config.with_features(REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES)
    preprocessor = MetadataPreprocessor(
        reduced_config,
        column_transforms={"anatomical_site": normalize_anatomical_site_for_cross_dataset},
    ).fit(train_df)
    return reduced_config, preprocessor


def train_one_run(dataset_name: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    train_df = pd.read_csv(ds_config.train_csv)
    reduced_config, preprocessor = build_reduced_preprocessor(ds_config, train_df)

    train_ds = MetadataDataset(reduced_config.train_csv, reduced_config, preprocessor)
    val_ds = MetadataDataset(reduced_config.val_csv, reduced_config, preprocessor)
    model = MetadataMLP(input_dim=preprocessor.output_dim, num_classes=reduced_config.num_classes).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        reduced_config.train_csv, reduced_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_METADATA, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"metadata_reduced_seed{seed}"
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
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": "metadata_reduced",
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
        "branch": "metadata_reduced",
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "numeric_features": REDUCED_NUMERIC_FEATURES,
        "categorical_features": REDUCED_CATEGORICAL_FEATURES,
        "metadata_input_dim": preprocessor.output_dim,
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 8 reduced-feature metadata training")
    parser.add_argument("--dataset", choices=["PAD_UFES20"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
