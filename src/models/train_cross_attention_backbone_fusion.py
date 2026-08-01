"""Step 4 training entrypoint - cross-attention fusion between metadata
and one of the two Phase 8B top-2 backbones (ConvNeXt-Tiny, DenseNet121).

Usage:
    python -m src.models.train_cross_attention_backbone_fusion \\
        --backbone convnext_tiny --seed 0
    python -m src.models.train_cross_attention_backbone_fusion \\
        --backbone densenet121 --seed 0

Dataset is NOT a CLI choice, unlike train_backbone_fusion.py - it is
always PAD_UFES20 (never PAD_UFES20_Expanded, which is
image_branch_only=True and structurally has no usable metadata for its
DERM12345/MED-NODE rows; see src/models/config.py's PAD_UFES20_EXPANDED
and Project_Tracking.md's "Step 2 Integration Plan").

Cross-dataset warm start, deliberately asymmetric:
  - image embedder <- PAD_UFES20_Expanded's Step 3 (Phase 8B) backbone
    checkpoint (logs/PAD_UFES20_Expanded/checkpoints/image_{backbone}_
    seed{N}_best.pt) - the larger image-only training set's benefit is
    already baked into these weights.
  - metadata embedder <- PAD_UFES20's own Phase 7 Stage 1 metadata
    checkpoint (ds_config.stage1_checkpoints_dir / metadata_seed{N}_
    best.pt) - the only dataset with real, non-NaN clinical metadata.
  - the train/val fine-tuning loop itself runs entirely on PAD_UFES20's
    original metadata_train.csv/metadata_val.csv - expanded rows are
    never seen here (they have no metadata to feed the metadata branch
    or the cross-attention mechanism), so this step sees zero expanded
    rows directly; their only influence is indirect, via the warm-started
    image weights.

Same loss/metric/seed/split discipline as every prior stage:
class-weighted CrossEntropyLoss, macro-F1 for model selection/early
stopping, SEEDS = [0, 1, 2], identical train/val split files. The test
split is never loaded here - only src/evaluation/evaluate.py reads it,
later, with --confirm-final.
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
from src.models.cross_attention_backbone_fusion_model import CrossAttentionBackboneFusionModel
from src.models.dataset import FusionDataset, MetadataPreprocessor
from src.models.spatial_backbone_embedder import SPATIAL_BACKBONE_NAMES
from src.models.train import _image_run_name, compute_class_weights, set_seed


def run_epoch_cross_attention_backbone(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, metadata, labels in loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
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
    return avg_loss, macro_f1


def train_one_run(backbone: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset("PAD_UFES20")
    expanded_ds_config = get_dataset("PAD_UFES20_Expanded")

    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
    train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)
    val_ds = FusionDataset(ds_config.val_csv, ds_config, preprocessor, train=False)

    model = CrossAttentionBackboneFusionModel(
        backbone_name=backbone,
        metadata_input_dim=preprocessor.output_dim,
        num_classes=ds_config.num_classes,
    ).to(device)

    image_run_name = _image_run_name(seed, backbone, sampler="shuffle", strong_augment="none")
    image_checkpoint = expanded_ds_config.checkpoints_dir / f"{image_run_name}_best.pt"
    metadata_checkpoint = ds_config.stage1_checkpoints_dir / f"metadata_seed{seed}_best.pt"
    for path in (image_checkpoint, metadata_checkpoint):
        if not path.exists():
            raise FileNotFoundError(
                f"Warm-start checkpoint not found: {path} - Step 4 requires "
                f"both the Step 3 (Phase 8B) PAD_UFES20_Expanded backbone "
                f"checkpoint and the Phase 7 Stage 1 PAD_UFES20 metadata "
                f"checkpoint to exist first."
            )
    model.load_warm_start_checkpoints(image_checkpoint, metadata_checkpoint, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_CROSS_ATTENTION, weight_decay=WEIGHT_DECAY
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"cross_attention_backbone_{backbone}_seed{seed}"
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
            train_loss, train_macro_f1 = run_epoch_cross_attention_backbone(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch_cross_attention_backbone(
                model, val_loader, criterion, optimizer, device, train=False
            )
            writer.writerow([epoch, train_loss, train_macro_f1, val_loss, val_macro_f1])
            f.flush()
            elapsed = time.time() - start
            print(
                f"[PAD_UFES20/{run_name}] epoch {epoch:02d} "
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
                        "dataset": "PAD_UFES20",
                        "branch": "cross_attention_backbone",
                        "backbone": backbone,
                        "seed": seed,
                        "epoch": epoch,
                        "val_macro_f1": val_macro_f1,
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

    summary = {
        "dataset": "PAD_UFES20",
        "branch": "cross_attention_backbone",
        "backbone": backbone,
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
        f"[PAD_UFES20/{run_name}] best val macro-F1: "
        f"{best_val_macro_f1:.4f} -> {checkpoint_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Step 4 cross-attention backbone fusion training (PAD_UFES20 only)"
    )
    parser.add_argument("--backbone", choices=SPATIAL_BACKBONE_NAMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.backbone, args.seed)


if __name__ == "__main__":
    main()
