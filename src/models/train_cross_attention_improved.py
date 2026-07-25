"""Score Improvement Experiments track - "best legitimate effort" combined
training run for the Phase 7 Stage 2 cross-attention fusion architecture.

Usage:
    python -m src.models.train_cross_attention_improved --dataset PAD_UFES20 --seed 0

Same warm-start-then-fine-tune scaffold as train_cross_attention_fusion.py
(warm-starts both embedders from Stage 1 checkpoints, fine-tunes end-to-end,
macro-F1 model selection/early stopping, identical seeds/splits) - this
script stacks every training-strategy improvement that does not touch
data/processed or data/raw into ONE combined run, rather than five separate
ablations:

1. Focal Loss (gamma=2.0, alpha = the same inverse-class-frequency weights
   compute_class_weights() already produces for the plain CrossEntropyLoss
   baseline), so hard/rare examples dominate the gradient more than easy
   ones.
2. WeightedRandomSampler - oversamples rare classes (Melanoma, SCC) during
   training so the model sees them more often per epoch. Real samples only,
   drawn with replacement; no synthetic data.
3. Stronger image augmentation (RandomRotation(30), RandomAffine(scale=
   (0.9, 1.1), translate=(0.05, 0.05)), on top of the existing flip/color
   jitter) - defined locally, dataset.py's build_image_transform is
   untouched since other scripts still rely on its milder recipe.
4. Label smoothing (0.1), passed straight into F.cross_entropy inside
   FocalLoss.
5. Cosine annealing LR schedule (max=LEARNING_RATE_CROSS_ATTENTION=1e-5,
   min=1e-7) over NUM_EPOCHS.

Whatever macro-F1 this run achieves is the final, reported number for this
architecture/dataset - after this, the project moves to Phase 8.

Checkpoints/logs use "cross_attention_improved" naming throughout, saved
alongside - never overwriting - the existing cross_attention_seed{N}_best.pt
checkpoints.
"""

import argparse
import csv
import json
import time

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.models.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    IMAGE_INPUT_SIZE,
    LEARNING_RATE_CROSS_ATTENTION,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    get_dataset,
)
from src.models.cross_attention_fusion_model import CrossAttentionFusionModel
from src.models.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    FusionDataset,
    MetadataPreprocessor,
    ResizePad,
)
from src.models.train import compute_class_weights, set_seed

LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
COSINE_ETA_MIN = 1e-7


class FocalLoss(torch.nn.Module):
    """Multi-class focal loss (Lin et al. 2017), class-weighted by `alpha`
    and combined with label smoothing (native F.cross_entropy support -
    applied to the one-hot target before the per-sample loss is computed,
    compatible with focal re-weighting since both operate per-sample).
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def build_stronger_train_transform() -> transforms.Compose:
    """Stronger augmentation than dataset.py's build_image_transform(train=True):
    adds RandomRotation(30) (up from 20) and RandomAffine(scale/translate),
    keeps the existing flip and color jitter. Defined here rather than in
    dataset.py so every other script's augmentation recipe stays unchanged.
    """
    return transforms.Compose([
        ResizePad(IMAGE_INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def compute_sample_weights(train_csv, class_names) -> list:
    """One weight per training row = 1 / (that row's class count), for
    WeightedRandomSampler - same inverse-frequency principle as
    compute_class_weights(), just expanded per-row instead of per-class-mean.
    """
    df = pd.read_csv(train_csv)
    counts = df["disease_label"].value_counts()
    return [1.0 / counts[label] for label in df["disease_label"]]


def run_epoch_cross_attention(model, loader, criterion, optimizer, device, train: bool):
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


def train_one_run(dataset_name: str, seed: int) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)

    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))
    train_ds = FusionDataset(ds_config.train_csv, ds_config, preprocessor, train=True)
    train_ds.transform = build_stronger_train_transform()
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
                f"warm-start requires both branches' Stage 1 checkpoints to "
                f"exist first."
            )
    model.load_stage1_checkpoints(image_checkpoint, metadata_checkpoint, device)

    sample_weights = compute_sample_weights(ds_config.train_csv, ds_config.class_names)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(
        ds_config.train_csv, ds_config.class_names
    ).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE_CROSS_ATTENTION, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=COSINE_ETA_MIN
    )

    ds_config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ds_config.logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"cross_attention_improved_seed{seed}"
    checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    metrics_csv_path = ds_config.logs_dir / f"train_{run_name}.csv"

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1", "lr"]
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            start = time.time()
            train_loss, train_macro_f1 = run_epoch_cross_attention(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_macro_f1 = run_epoch_cross_attention(
                model, val_loader, criterion, optimizer, device, train=False
            )
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            writer.writerow(
                [epoch, train_loss, train_macro_f1, val_loss, val_macro_f1, current_lr]
            )
            f.flush()
            elapsed = time.time() - start
            print(
                f"[{dataset_name}/{run_name}] epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_macroF1={train_macro_f1:.4f} "
                f"val_loss={val_loss:.4f} val_macroF1={val_macro_f1:.4f} "
                f"lr={current_lr:.2e} ({elapsed:.1f}s)"
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "dataset": dataset_name,
                        "branch": "cross_attention_improved",
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
                    print(f"[{dataset_name}/{run_name}] early stopping at epoch {epoch}")
                    break

    summary = {
        "dataset": dataset_name,
        "branch": "cross_attention_improved",
        "seed": seed,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "warm_start_image_checkpoint": str(image_checkpoint),
        "warm_start_metadata_checkpoint": str(metadata_checkpoint),
        "techniques": {
            "focal_loss_gamma": FOCAL_GAMMA,
            "label_smoothing": LABEL_SMOOTHING,
            "weighted_random_sampler": True,
            "stronger_augmentation": True,
            "cosine_annealing_lr": {
                "max": LEARNING_RATE_CROSS_ATTENTION,
                "min": COSINE_ETA_MIN,
            },
        },
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
        description="Score Improvement Experiments - combined best-effort cross-attention training"
    )
    parser.add_argument("--dataset", choices=["PAD_UFES20"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    train_one_run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
