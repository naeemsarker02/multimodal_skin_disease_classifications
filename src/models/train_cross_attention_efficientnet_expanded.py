"""Dataset-expansion-only ablation (approved 2026-08-01, Project_Tracking.md
"Pre-Registered Prediction - Dataset-Expansion-Only Ablation") - isolates
whether PAD_UFES20_Expanded's added training images help the ORIGINAL
Phase 7 Stage 2 architecture (CrossAttentionFusionModel, EfficientNet-B0
image embedder), holding architecture constant, separate from Step 4's
backbone-architecture-change effect (ConvNeXt-Tiny/DenseNet121).

Usage:
    python -m src.models.train_cross_attention_efficientnet_expanded --seed 0

Same warm-start-then-fine-tune pattern as Step 4's
train_cross_attention_backbone_fusion.py, just with the original
(unparameterized) CrossAttentionFusionModel/SpatialImageEmbedder instead
of the backbone-parameterized CrossAttentionBackboneFusionModel:
- Image embedder warm-started from Step 3's (Phase 8B)
  PAD_UFES20_Expanded EfficientNet-B0 checkpoint
  (logs/PAD_UFES20_Expanded/checkpoints/image_seed{N}_best.pt) - the
  larger image-only training set's benefit is transferred in via this
  checkpoint only.
- Metadata embedder warm-started from PAD_UFES20's own Phase 7 Stage 1
  checkpoint (logs/PAD_UFES20/checkpoints/metadata_seed{N}_best.pt) -
  identical to every prior cross-attention variant.
- The fine-tuning loop itself trains/validates on PAD_UFES20's ORIGINAL
  metadata_train.csv/metadata_val.csv (not the expanded CSV) - expanded
  rows have no compatible metadata (image_branch_only=True), so this
  script never reads PAD_UFES20_Expanded's metadata at all, exactly like
  Step 4.

Naming-collision safety (verified before this file was written - see
Project_Tracking.md "PAD-UFES-20 Test-Split Guard..." session's naming
check): run_name/checkpoint/summary/csv all use the
"cross_attention_efficientnet_expanded" prefix, distinct from both the
locked "cross_attention_seed{N}_best.pt" (Phase 7 Stage 2, 0.6977 test)
and Step 4's "cross_attention_backbone_{convnext_tiny,densenet121}_seed{N}_best.pt".
Confirmed via grep that this exact string was unused anywhere in the
repo before this file existed.

Test split is never read here (only src/evaluation/evaluate.py reads
test, and only with --confirm-final - separately still blocked twice
over by test_split_guard.py's marker for PAD_UFES20).
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
from src.models.train import _image_run_name, compute_class_weights, set_seed

RUN_PREFIX = "cross_attention_efficientnet_expanded"


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
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


def train_one_run(seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset("PAD_UFES20")
    expanded_ds_config = get_dataset("PAD_UFES20_Expanded")

    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
    train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)
    val_ds = FusionDataset(ds_config.val_csv, ds_config, preprocessor, train=False)

    model = CrossAttentionFusionModel(
        metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
    ).to(device)

    image_run_name = _image_run_name(seed, backbone="efficientnet_b0", sampler="shuffle", strong_augment="none")
    image_checkpoint = expanded_ds_config.checkpoints_dir / f"{image_run_name}_best.pt"
    metadata_checkpoint = ds_config.stage1_checkpoints_dir / f"metadata_seed{seed}_best.pt"
    for path in (image_checkpoint, metadata_checkpoint):
        if not path.exists():
            raise FileNotFoundError(
                f"Warm-start checkpoint not found: {path} - this ablation requires "
                f"both the Step 3 (Phase 8B) PAD_UFES20_Expanded EfficientNet-B0 "
                f"image checkpoint and the Phase 7 Stage 1 PAD_UFES20 metadata "
                f"checkpoint to exist first."
            )
    model.load_stage1_checkpoints(image_checkpoint, metadata_checkpoint, device)

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
    run_name = f"{RUN_PREFIX}_seed{seed}"
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
                        "branch": RUN_PREFIX,
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
        "branch": RUN_PREFIX,
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
        description="Dataset-expansion-only ablation: original cross-attention "
        "architecture (EfficientNet-B0), warm-started from the expanded-dataset "
        "image checkpoint (PAD_UFES20 only)"
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.seed)


if __name__ == "__main__":
    main()
