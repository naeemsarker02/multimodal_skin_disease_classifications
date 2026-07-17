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
from torch.utils.data import DataLoader

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_IMAGE,
    LEARNING_RATE_METADATA,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    get_dataset,
)
from src.models.dataset import ImageDataset, MetadataDataset, MetadataPreprocessor
from src.models.image_model import build_efficientnet_b0
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


def train_one_run(dataset_name: str, branch: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    if branch == "image":
        train_ds = ImageDataset(ds_config.train_csv, ds_config, train=True)
        val_ds = ImageDataset(ds_config.val_csv, ds_config, train=False)
        model = build_efficientnet_b0(num_classes=ds_config.num_classes).to(device)
        lr = LEARNING_RATE_IMAGE
    elif branch == "metadata":
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
        train_ds = MetadataDataset(ds_config.train_csv, ds_config, preprocessor)
        val_ds = MetadataDataset(ds_config.val_csv, ds_config, preprocessor)
        model = MetadataMLP(
            input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        ).to(device)
        lr = LEARNING_RATE_METADATA
    else:
        raise ValueError(f"Unknown branch: {branch}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{branch}_seed{seed}"
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
                        "branch": branch,
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
                        "num_classes": ds_config.num_classes,
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
        "branch": branch,
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Stage 1 training")
    parser.add_argument("--dataset", choices=["PAD_UFES20", "HAM10000"], required=True)
    parser.add_argument("--branch", choices=["image", "metadata"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.branch, args.seed)


if __name__ == "__main__":
    main()
