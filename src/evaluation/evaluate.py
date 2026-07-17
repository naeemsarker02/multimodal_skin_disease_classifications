"""Phase 6/7 Stage 1 evaluation entrypoint - dataset-parameterized
baselines plus the PAD-UFES-20-only fusion branch.

Usage:
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch image --seed 0 --split val
    python -m src.evaluation.evaluate --dataset HAM10000 --branch image --seed 0 --split test --confirm-final
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch fusion --seed 0 --split val

This is the only script that reads metadata_test.csv. Evaluating against
--split test requires --confirm-final, a deliberate friction point so the
test split cannot be touched by accident mid-tuning - it should only be
used once, after all training/model-selection decisions (Stage 1) are
already finalized. See Project_Tracking.md decision (4).
"""

import argparse
import json

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.models.config import BATCH_SIZE, get_dataset
from src.models.dataset import (
    FusionDataset,
    ImageDataset,
    MetadataDataset,
    MetadataPreprocessor,
)
from src.models.fusion_model import FusionModel
from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP


def load_model(branch: str, checkpoint_path, ds_config, device, preprocessor=None):
    if branch == "image":
        model = build_efficientnet_b0(num_classes=ds_config.num_classes)
    elif branch == "metadata":
        model = MetadataMLP(
            input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        )
    elif branch == "fusion":
        model = FusionModel(
            metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        )
    else:
        raise ValueError(f"Unknown branch: {branch}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_eval_dataset(branch: str, csv_path, ds_config, preprocessor=None):
    if branch == "image":
        return ImageDataset(csv_path, ds_config, train=False)
    if branch == "metadata":
        return MetadataDataset(csv_path, ds_config, preprocessor)
    if branch == "fusion":
        return FusionDataset(csv_path, ds_config, preprocessor, train=False)
    raise ValueError(f"Unknown branch: {branch}")


def evaluate(dataset_name: str, branch: str, seed: int, split: str) -> dict:
    if branch == "fusion" and dataset_name != "PAD_UFES20":
        raise ValueError(
            "Fusion checkpoints only exist for PAD_UFES20 (Phase 7 Stage 1 "
            "scope) - not trained/evaluated on any other dataset."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)
    checkpoint_path = ds_config.checkpoints_dir / f"{branch}_seed{seed}_best.pt"

    preprocessor = None
    if branch in ("metadata", "fusion"):
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))

    model = load_model(branch, checkpoint_path, ds_config, device, preprocessor)

    split_csv = {"val": ds_config.val_csv, "test": ds_config.test_csv}[split]
    dataset = build_eval_dataset(branch, split_csv, ds_config, preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if branch == "fusion":
                images, metadata, labels = batch
                outputs = model(images.to(device), metadata.to(device))
            else:
                inputs, labels = batch
                outputs = model(inputs.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    class_names = ds_config.class_names
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(len(class_names))),
        zero_division=0,
    )
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    result = {
        "dataset": dataset_name,
        "branch": branch,
        "seed": seed,
        "split": split,
        "macro_f1": macro_f1,
        "accuracy_reference_only": accuracy,
        "per_class_f1": dict(zip(class_names, per_class_f1.tolist())),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_class_order": class_names,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 6/7 Stage 1 evaluation")
    parser.add_argument("--dataset", choices=["PAD_UFES20", "HAM10000"], required=True)
    parser.add_argument("--branch", choices=["image", "metadata", "fusion"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Required to evaluate on --split test. Only pass this once all "
        "Stage 1 training/model-selection decisions are finalized.",
    )
    args = parser.parse_args()

    if args.branch == "fusion" and args.dataset != "PAD_UFES20":
        raise SystemExit(
            "--branch fusion is only valid with --dataset PAD_UFES20 (Phase 7 "
            "Stage 1 scope - see Project_Tracking.md)."
        )

    if args.split == "test" and not args.confirm_final:
        raise SystemExit(
            "Refusing to evaluate on the test split without --confirm-final. "
            "The test split must stay untouched until the final Stage 1 "
            "evaluation run (see Project_Tracking.md decision 4)."
        )

    result = evaluate(args.dataset, args.branch, args.seed, args.split)

    ds_config = get_dataset(args.dataset)
    reports_dir = (
        ds_config.fusion_reports_dir if args.branch == "fusion" else ds_config.baseline_reports_dir
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"eval_{args.branch}_seed{args.seed}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"macro-F1 ({args.split}): {result['macro_f1']:.4f}")
    print("per-class F1:")
    for name, score in result["per_class_f1"].items():
        print(f"  {name}: {score:.4f}")
    print(f"accuracy (reference only): {result['accuracy_reference_only']:.4f}")
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
