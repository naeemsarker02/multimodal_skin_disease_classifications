"""Phase 6/7 Stage 1 evaluation entrypoint - dataset-parameterized
baselines plus the PAD-UFES-20-only fusion branch.

Usage:
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch image --seed 0 --split val
    python -m src.evaluation.evaluate --dataset HAM10000 --branch image --seed 0 --split test --confirm-final
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch fusion --seed 0 --split val
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch cross_attention --seed 0 --split val
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch cross_attention --ensemble-seeds 0,1,2 --split val

This is the only script that reads metadata_test.csv. Evaluating against
--split test requires --confirm-final, a deliberate friction point so the
test split cannot be touched by accident mid-tuning - it should only be
used once, after all training/model-selection decisions (Stage 1) are
already finalized. See Project_Tracking.md decision (4).

The cross_attention branch and --ensemble-seeds were added for the Score
Improvement Experiments track (separate from Phase 8's in-progress work,
never touches the test split) - additive only, the existing
image/metadata/fusion single-checkpoint eval path is unchanged.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.models.config import BATCH_SIZE, get_dataset
from src.models.cross_attention_fusion_model import CrossAttentionFusionModel
from src.models.dataset import (
    FusionDataset,
    ImageDataset,
    MetadataDataset,
    MetadataPreprocessor,
)
from src.models.fusion_model import FusionModel
from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP
from src.evaluation.test_split_guard import check_test_split_available


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
    elif branch in ("cross_attention", "cross_attention_improved"):
        model = CrossAttentionFusionModel(
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
    if branch in ("fusion", "cross_attention", "cross_attention_improved"):
        return FusionDataset(csv_path, ds_config, preprocessor, train=False)
    raise ValueError(f"Unknown branch: {branch}")


def evaluate(dataset_name: str, branch: str, seed: int, split: str) -> dict:
    if branch in ("fusion", "cross_attention", "cross_attention_improved") and dataset_name != "PAD_UFES20":
        raise ValueError(
            "Fusion/cross-attention checkpoints only exist for PAD_UFES20 "
            "(Phase 7 scope) - not trained/evaluated on any other dataset."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)
    if split == "test":
        check_test_split_available(ds_config, "evaluate.py")
    checkpoint_path = ds_config.checkpoints_dir / f"{branch}_seed{seed}_best.pt"

    preprocessor = None
    if branch in ("metadata", "fusion", "cross_attention", "cross_attention_improved"):
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))

    model = load_model(branch, checkpoint_path, ds_config, device, preprocessor)

    split_csv = {"val": ds_config.val_csv, "test": ds_config.test_csv}[split]
    dataset = build_eval_dataset(branch, split_csv, ds_config, preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if branch in ("fusion", "cross_attention"):
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


def evaluate_ensemble(
    dataset_name: str, branch: str, seeds: list, split: str, tta: bool = False
) -> dict:
    """Score Improvement Experiments track: average softmax across multiple
    seed checkpoints of the same branch (currently cross_attention only -
    the 3 already-trained Phase 7 Stage 2 checkpoints), then argmax. No
    retraining involved. Kept fully separate from evaluate() above so the
    existing single-checkpoint branches are untouched.

    tta=True adds horizontal-flip test-time augmentation: each model's
    softmax is itself averaged over the original image and its horizontal
    flip before being folded into the cross-seed average, so with 3 seeds
    this becomes a 2 views x 3 seeds = 6-way softmax average. Flip is
    applied directly to the already-resized/normalized tensor
    (torch.flip on the width axis) - no new Dataset/transform needed since
    horizontal mirroring commutes with resize/normalize.
    """
    if branch != "cross_attention":
        raise ValueError(
            "Ensemble evaluation is currently only supported for "
            "branch='cross_attention'."
        )
    if dataset_name != "PAD_UFES20":
        raise ValueError(
            "cross_attention checkpoints only exist for PAD_UFES20 (Phase 7 scope)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)
    if split == "test":
        check_test_split_available(ds_config, "evaluate.py (ensemble)")
    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))

    models = []
    for seed in seeds:
        checkpoint_path = ds_config.checkpoints_dir / f"{branch}_seed{seed}_best.pt"
        models.append(load_model(branch, checkpoint_path, ds_config, device, preprocessor))

    split_csv = {"val": ds_config.val_csv, "test": ds_config.test_csv}[split]
    dataset = build_eval_dataset(branch, split_csv, ds_config, preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    per_model_probs = [[] for _ in models]
    all_labels = []
    with torch.no_grad():
        for images, metadata, labels in loader:
            images = images.to(device)
            metadata = metadata.to(device)
            for i, model in enumerate(models):
                outputs = model(images, metadata)
                probs = torch.softmax(outputs, dim=1)
                if tta:
                    flipped = torch.flip(images, dims=[3])
                    outputs_flipped = model(flipped, metadata)
                    probs_flipped = torch.softmax(outputs_flipped, dim=1)
                    probs = (probs + probs_flipped) / 2
                per_model_probs[i].append(probs.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

    stacked = np.stack(
        [np.concatenate(probs, axis=0) for probs in per_model_probs], axis=0
    )  # [num_models, N, num_classes]
    avg_probs = stacked.mean(axis=0)
    all_preds = avg_probs.argmax(axis=1).tolist()

    class_names = ds_config.class_names
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(len(class_names))),
        zero_division=0,
    )
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    return {
        "dataset": dataset_name,
        "branch": branch,
        "seeds": seeds,
        "ensemble": True,
        "tta": tta,
        "split": split,
        "macro_f1": macro_f1,
        "accuracy_reference_only": accuracy,
        "per_class_f1": dict(zip(class_names, per_class_f1.tolist())),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_class_order": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6/7 Stage 1 evaluation")
    parser.add_argument("--dataset", choices=["PAD_UFES20", "HAM10000"], required=True)
    parser.add_argument(
        "--branch",
        choices=["image", "metadata", "fusion", "cross_attention", "cross_attention_improved"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--ensemble-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to ensemble via softmax averaging, e.g. "
        "'0,1,2'. Score Improvement Experiments track only - currently "
        "requires --branch cross_attention. Mutually exclusive with --seed.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Horizontal-flip test-time augmentation (2-way softmax average, "
        "or 2 x num-ensemble-seeds if combined with --ensemble-seeds). "
        "Score Improvement Experiments track only - currently requires "
        "--branch cross_attention and --ensemble-seeds.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Required to evaluate on --split test. Only pass this once all "
        "Stage 1 training/model-selection decisions are finalized.",
    )
    args = parser.parse_args()

    if args.branch in ("fusion", "cross_attention", "cross_attention_improved") and args.dataset != "PAD_UFES20":
        raise SystemExit(
            f"--branch {args.branch} is only valid with --dataset PAD_UFES20 "
            "(Phase 7 scope - see Project_Tracking.md)."
        )

    if args.split == "test" and not args.confirm_final:
        raise SystemExit(
            "Refusing to evaluate on the test split without --confirm-final. "
            "The test split must stay untouched until the final Stage 1 "
            "evaluation run (see Project_Tracking.md decision 4)."
        )

    if args.ensemble_seeds is not None and args.seed is not None:
        raise SystemExit("Pass either --seed or --ensemble-seeds, not both.")
    if args.ensemble_seeds is None and args.seed is None:
        raise SystemExit("One of --seed or --ensemble-seeds is required.")
    if args.ensemble_seeds is not None and args.branch != "cross_attention":
        raise SystemExit("--ensemble-seeds currently requires --branch cross_attention.")
    if args.tta and args.ensemble_seeds is None:
        raise SystemExit("--tta currently requires --ensemble-seeds too.")

    ds_config = get_dataset(args.dataset)

    if args.ensemble_seeds is not None:
        seeds = [int(s) for s in args.ensemble_seeds.split(",")]
        result = evaluate_ensemble(args.dataset, args.branch, seeds, args.split, tta=args.tta)
        reports_dir = ds_config.baseline_reports_dir.parent / "score_experiments"
        tta_suffix = "_tta" if args.tta else ""
        out_name = f"eval_{args.branch}_ensemble{len(seeds)}{tta_suffix}_{args.split}.json"
    else:
        result = evaluate(args.dataset, args.branch, args.seed, args.split)
        if args.branch == "fusion":
            reports_dir = ds_config.fusion_reports_dir
        elif args.branch in ("cross_attention", "cross_attention_improved"):
            reports_dir = ds_config.baseline_reports_dir.parent / "score_experiments"
        else:
            reports_dir = ds_config.baseline_reports_dir
        out_name = f"eval_{args.branch}_seed{args.seed}_{args.split}.json"

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / out_name
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
