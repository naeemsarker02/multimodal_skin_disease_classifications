"""Phase 8B fusion training entrypoint - PAD-UFES-20 only, top-2 backbones
from the 5-backbone comparison (src/models/train.py --backbone).

Usage:
    python -m src.models.train_backbone_fusion --dataset PAD_UFES20 \\
        --backbone-a resnet50 --backbone-b densenet121 --seed 0

If Step 3a's imbalance ablations (train.py --sampler / --strong-augment)
were adopted, pass the same --sampler/--strong-augment values used to train
the top-2 backbones' Step 2 checkpoints, so this warm-starts from the right
checkpoint filenames (src/models/train.py's _image_run_name naming scheme).

Warm-starts both embedders from their own Step 2 checkpoints (seed-matched,
same convention as train_fusion.py), then fine-tunes the whole model
end-to-end at LEARNING_RATE_FUSION - lower than LEARNING_RATE_IMAGE, since
both backbones are already converged. Same loss/metric/seed/split discipline
as train.py and train_fusion.py: class-weighted CrossEntropyLoss, macro-F1
for model selection/early stopping, SEEDS = [0, 1, 2], identical train/val
split files. The test split is never loaded here - only
src/evaluation/evaluate.py reads it, later, with --confirm-final.
"""

import argparse
import csv
import json
import time

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.backbone_fusion_model import BackboneFusionModel
from src.models.backbones import BACKBONE_NAMES
from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE_FUSION,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    get_dataset,
)
from src.models.dataset import ImageDataset
from src.models.train import _image_run_name, compute_class_weights, set_seed


def run_epoch_backbone_fusion(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
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


def train_one_run(
    dataset_name: str,
    backbone_a: str,
    backbone_b: str,
    seed: int,
    sampler: str = "shuffle",
    strong_augment: str = "none",
) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    train_ds = ImageDataset(ds_config.train_csv, ds_config, train=True)
    val_ds = ImageDataset(ds_config.val_csv, ds_config, train=False)

    model = BackboneFusionModel(
        backbone_a=backbone_a, backbone_b=backbone_b, num_classes=ds_config.num_classes
    ).to(device)

    checkpoint_a = ds_config.checkpoints_dir / (
        _image_run_name(seed, backbone_a, sampler, strong_augment) + "_best.pt"
    )
    checkpoint_b = ds_config.checkpoints_dir / (
        _image_run_name(seed, backbone_b, sampler, strong_augment) + "_best.pt"
    )
    for path in (checkpoint_a, checkpoint_b):
        if not path.exists():
            raise FileNotFoundError(
                f"Step 2 backbone checkpoint not found: {path} - fusion warm-start "
                f"requires both backbones' Step 2 checkpoints to exist first "
                f"(run train.py --branch image --backbone ... first)."
            )
    model.load_stage_checkpoints(checkpoint_a, checkpoint_b, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_FUSION, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"backbone_fusion_{backbone_a}_{backbone_b}_seed{seed}"
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
            train_loss, train_macro_f1 = run_epoch_backbone_fusion(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch_backbone_fusion(
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
                        "branch": "backbone_fusion",
                        "backbone_a": backbone_a,
                        "backbone_b": backbone_b,
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
        "branch": "backbone_fusion",
        "backbone_a": backbone_a,
        "backbone_b": backbone_b,
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "warm_start_checkpoint_a": str(checkpoint_a),
        "warm_start_checkpoint_b": str(checkpoint_b),
    }
    summary_path = ds_config.logs_dir / f"train_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[{dataset_name}/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 8B backbone fusion training")
    parser.add_argument("--dataset", choices=["PAD_UFES20"], required=True)
    parser.add_argument("--backbone-a", choices=BACKBONE_NAMES, required=True, dest="backbone_a")
    parser.add_argument("--backbone-b", choices=BACKBONE_NAMES, required=True, dest="backbone_b")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sampler", choices=["shuffle", "weighted"], default="shuffle")
    parser.add_argument(
        "--strong-augment", choices=["none", "minority"], default="none", dest="strong_augment"
    )
    args = parser.parse_args()
    if args.backbone_a == args.backbone_b:
        parser.error("--backbone-a and --backbone-b must be the top 2 *distinct* backbones")
    train_one_run(
        args.dataset, args.backbone_a, args.backbone_b, args.seed,
        sampler=args.sampler, strong_augment=args.strong_augment,
    )


if __name__ == "__main__":
    main()
