"""Phase 6/7 Stage 1 evaluation entrypoint - dataset-parameterized
baselines plus the PAD-UFES-20-only fusion branch.

Usage:
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch image --seed 0 --split val
    python -m src.evaluation.evaluate --dataset HAM10000 --branch image --seed 0 --split test --confirm-final
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch fusion --seed 0 --split val
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch cross_attention --seed 0 --split val
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch cross_attention --ensemble-seeds 0,1,2 --split val
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch cross_attention_backbone --backbone convnext_tiny --seed 0 --split val
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch cross_attention_backbone_ensemble --seed 0 --split val
    python -m src.evaluation.evaluate --dataset PAD_UFES20 --branch cross_attention_efficientnet_expanded --seed 0 --split val

This is the only script that reads metadata_test.csv. Evaluating against
--split test requires --confirm-final, a deliberate friction point so the
test split cannot be touched by accident mid-tuning - it should only be
used once, after all training/model-selection decisions (Stage 1) are
already finalized. See Project_Tracking.md decision (4).

The cross_attention branch and --ensemble-seeds were added for the Score
Improvement Experiments track (separate from Phase 8's in-progress work,
never touches the test split) - additive only, the existing
image/metadata/fusion single-checkpoint eval path is unchanged.

cross_attention_backbone (Step 4, PAD_UFES20 only) evaluates a single
CrossAttentionBackboneFusionModel checkpoint (--backbone convnext_tiny or
densenet121, --seed required). cross_attention_backbone_ensemble
combines the two backbones' seed-matched checkpoints via unweighted
softmax-probability averaging (no training, no learned combiner - see
Project_Tracking.md's Step 4 plan for why: every split already has a
fixed role - train fits the base models, val selects/early-stops them,
test is locked - so a parameter-free combination step is the only one
that doesn't need a new place to be fit without leaking one of those
roles).

cross_attention_efficientnet_expanded (dataset-expansion-only ablation,
approved 2026-08-01, PAD_UFES20 only) evaluates the ORIGINAL
CrossAttentionFusionModel (EfficientNet-B0) architecture, warm-started
from PAD_UFES20_Expanded's image checkpoint instead of PAD_UFES20's own
- isolates the dataset-expansion effect from Step 4's backbone-change
effect. Validation-only by design: --split test is refused for this
branch unconditionally (see Project_Tracking.md 'Pre-Registered
Prediction'), independent of PAD_UFES20's test_split_guard.py marker
(already twice-consumed regardless).
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.models.backbone_fusion_model import BackboneFusionModel
from src.models.backbones import BACKBONE_NAMES, build_backbone
from src.models.config import BATCH_SIZE, DATASETS, get_dataset
from src.models.cross_attention_backbone_fusion_model import CrossAttentionBackboneFusionModel
from src.models.cross_attention_fusion_model import CrossAttentionFusionModel
from src.models.dataset import (
    FusionDataset,
    ImageDataset,
    MetadataDataset,
    MetadataPreprocessor,
)
from src.models.fusion_model import FusionModel
from src.models.metadata_model import MetadataMLP
from src.models.spatial_backbone_embedder import SPATIAL_BACKBONE_NAMES
from src.models.train import _image_run_name
from src.evaluation.test_split_guard import check_test_split_available


def load_model(branch: str, checkpoint_path, ds_config, device, preprocessor=None,
                backbone: str = "efficientnet_b0"):
    # Loaded up front (rather than after model construction, like the other
    # branches below) because branch="backbone_fusion" needs the checkpoint's
    # own backbone_a/backbone_b fields to know which two architectures to
    # build before load_state_dict can run.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if branch == "image":
        model = build_backbone(backbone, num_classes=ds_config.num_classes)
    elif branch == "metadata":
        model = MetadataMLP(
            input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        )
    elif branch == "fusion":
        model = FusionModel(
            metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        )
    elif branch in ("cross_attention", "cross_attention_improved", "cross_attention_efficientnet_expanded"):
        model = CrossAttentionFusionModel(
            metadata_input_dim=preprocessor.output_dim, num_classes=ds_config.num_classes
        )
    elif branch == "backbone_fusion":
        model = BackboneFusionModel(
            checkpoint["backbone_a"], checkpoint["backbone_b"],
            num_classes=ds_config.num_classes,
        )
    elif branch == "cross_attention_backbone":
        # backbone name read from the checkpoint itself (like backbone_fusion
        # above), not from the outer --backbone CLI arg - avoids a mismatch
        # between a checkpoint's actual architecture and whatever --backbone
        # value the caller happened to pass.
        model = CrossAttentionBackboneFusionModel(
            checkpoint["backbone"], metadata_input_dim=preprocessor.output_dim,
            num_classes=ds_config.num_classes,
        )
    else:
        raise ValueError(f"Unknown branch: {branch}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_eval_dataset(branch: str, csv_path, ds_config, preprocessor=None):
    if branch in ("image", "backbone_fusion"):
        return ImageDataset(csv_path, ds_config, train=False)
    if branch == "metadata":
        return MetadataDataset(csv_path, ds_config, preprocessor)
    if branch in (
        "fusion", "cross_attention", "cross_attention_improved",
        "cross_attention_backbone", "cross_attention_efficientnet_expanded",
    ):
        return FusionDataset(csv_path, ds_config, preprocessor, train=False)
    raise ValueError(f"Unknown branch: {branch}")


def evaluate(
    dataset_name: str, branch: str, seed: int, split: str,
    backbone: str = "efficientnet_b0", sampler: str = "shuffle",
    strong_augment: str = "none", backbone_a: str = None, backbone_b: str = None,
) -> dict:
    if branch in (
        "fusion", "cross_attention", "cross_attention_improved",
        "backbone_fusion", "cross_attention_backbone",
        "cross_attention_efficientnet_expanded",
    ) and dataset_name != "PAD_UFES20":
        raise ValueError(
            "Fusion/cross-attention/backbone_fusion checkpoints only exist for "
            "PAD_UFES20 (Phase 7/8B/Step 4 scope) - not trained/evaluated on any other dataset."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_config = get_dataset(dataset_name)
    if split == "test":
        check_test_split_available(ds_config, "evaluate.py")

    if branch == "image":
        run_name = _image_run_name(seed, backbone, sampler, strong_augment)
        checkpoint_path = ds_config.checkpoints_dir / f"{run_name}_best.pt"
    elif branch == "backbone_fusion":
        if backbone_a is None or backbone_b is None:
            raise ValueError("branch='backbone_fusion' requires backbone_a and backbone_b.")
        checkpoint_path = (
            ds_config.checkpoints_dir
            / f"backbone_fusion_{backbone_a}_{backbone_b}_seed{seed}_best.pt"
        )
    elif branch == "cross_attention_backbone":
        checkpoint_path = (
            ds_config.checkpoints_dir / f"cross_attention_backbone_{backbone}_seed{seed}_best.pt"
        )
    else:
        checkpoint_path = ds_config.checkpoints_dir / f"{branch}_seed{seed}_best.pt"

    preprocessor = None
    if branch in (
        "metadata", "fusion", "cross_attention", "cross_attention_improved",
        "cross_attention_backbone", "cross_attention_efficientnet_expanded",
    ):
        preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))

    model = load_model(branch, checkpoint_path, ds_config, device, preprocessor, backbone=backbone)

    split_csv = {"val": ds_config.val_csv, "test": ds_config.test_csv}[split]
    dataset = build_eval_dataset(branch, split_csv, ds_config, preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if branch in (
                "fusion", "cross_attention", "cross_attention_backbone",
                "cross_attention_efficientnet_expanded",
            ):
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
    if branch == "cross_attention_backbone":
        # Row order matches dataset.df (loader uses shuffle=False) - needed to
        # pair-resample against the locked cross_attention (EfficientNet-B0)
        # test predictions in bootstrap_significance_backbone_fusion.py.
        predictions_df = dataset.df.copy()
        predictions_df["true_label_idx"] = all_labels
        predictions_df["pred_label_idx"] = all_preds
        result["_predictions_df"] = predictions_df
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


def evaluate_dual_backbone_ensemble(
    seed: int, split: str, backbone_a: str = "convnext_tiny", backbone_b: str = "densenet121"
) -> dict:
    """Step 4's approved ensemble mechanism: unweighted softmax-probability
    averaging between the two seed-matched cross_attention_backbone
    checkpoints (ConvNeXt-Tiny + DenseNet121, the Phase 8B top-2). No
    training, no learned combiner - see this module's docstring for why
    (every split already has a fixed role; a parameter-free combination
    step is the only one that needs no new place to be fit).

    Distinct from evaluate_ensemble() above, which averages across seeds
    of the *same* branch/backbone - this averages across the two
    *different* backbones at the *same* seed.
    """
    ds_config = get_dataset("PAD_UFES20")
    if split == "test":
        check_test_split_available(ds_config, "evaluate.py (dual-backbone ensemble)")
    preprocessor = MetadataPreprocessor(ds_config).fit(pd.read_csv(ds_config.train_csv))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for backbone in (backbone_a, backbone_b):
        checkpoint_path = (
            ds_config.checkpoints_dir / f"cross_attention_backbone_{backbone}_seed{seed}_best.pt"
        )
        models.append(
            load_model("cross_attention_backbone", checkpoint_path, ds_config, device, preprocessor)
        )

    split_csv = {"val": ds_config.val_csv, "test": ds_config.test_csv}[split]
    dataset = build_eval_dataset("cross_attention_backbone", split_csv, ds_config, preprocessor)
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
                per_model_probs[i].append(probs.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

    stacked = np.stack(
        [np.concatenate(probs, axis=0) for probs in per_model_probs], axis=0
    )  # [2, N, num_classes]
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

    result = {
        "dataset": "PAD_UFES20",
        "branch": "cross_attention_backbone_ensemble",
        "backbone_a": backbone_a,
        "backbone_b": backbone_b,
        "seed": seed,
        "ensemble": True,
        "split": split,
        "macro_f1": macro_f1,
        "accuracy_reference_only": accuracy,
        "per_class_f1": dict(zip(class_names, per_class_f1.tolist())),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_class_order": class_names,
    }
    # Row order matches dataset.df (loader uses shuffle=False) - needed to
    # pair-resample against the locked cross_attention (EfficientNet-B0)
    # test predictions in bootstrap_significance_backbone_fusion.py.
    predictions_df = dataset.df.copy()
    predictions_df["true_label_idx"] = all_labels
    predictions_df["pred_label_idx"] = all_preds
    result["_predictions_df"] = predictions_df
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 6/7 Stage 1 evaluation")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument(
        "--branch",
        choices=[
            "image", "metadata", "fusion", "cross_attention",
            "cross_attention_improved", "backbone_fusion",
            "cross_attention_backbone", "cross_attention_backbone_ensemble",
            "cross_attention_efficientnet_expanded",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--backbone", choices=BACKBONE_NAMES, default="efficientnet_b0",
        help="Phase 8B 5-backbone comparison (--branch image only). Must match "
             "the --backbone the checkpoint was trained with.",
    )
    parser.add_argument(
        "--sampler", choices=["shuffle", "weighted"], default="shuffle",
        help="Step 3a ablation (a) (--branch image only). Must match training.",
    )
    parser.add_argument(
        "--strong-augment", choices=["none", "minority"], default="none",
        dest="strong_augment",
        help="Step 3a ablation (b) (--branch image only). Must match training.",
    )
    parser.add_argument(
        "--backbone-a", default=None, dest="backbone_a",
        help="Phase 8B --branch backbone_fusion only: first backbone name "
             "(matches the checkpoint filename written by train_backbone_fusion.py).",
    )
    parser.add_argument(
        "--backbone-b", default=None, dest="backbone_b",
        help="Phase 8B --branch backbone_fusion only: second backbone name.",
    )
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

    if args.branch in (
        "fusion", "cross_attention", "cross_attention_improved", "backbone_fusion",
        "cross_attention_backbone", "cross_attention_backbone_ensemble",
        "cross_attention_efficientnet_expanded",
    ) and args.dataset != "PAD_UFES20":
        raise SystemExit(
            f"--branch {args.branch} is only valid with --dataset PAD_UFES20 "
            "(Phase 7/8B/Step 4 scope - see Project_Tracking.md)."
        )
    if args.branch == "backbone_fusion" and (args.backbone_a is None or args.backbone_b is None):
        raise SystemExit("--branch backbone_fusion requires --backbone-a and --backbone-b.")
    if args.branch == "cross_attention_backbone" and args.backbone not in SPATIAL_BACKBONE_NAMES:
        raise SystemExit(
            f"--branch cross_attention_backbone requires --backbone in {SPATIAL_BACKBONE_NAMES}."
        )
    if args.branch == "cross_attention_efficientnet_expanded" and args.split == "test":
        raise SystemExit(
            "--branch cross_attention_efficientnet_expanded is a validation-only "
            "exploratory ablation (dataset-expansion-only, approved 2026-08-01 - "
            "see Project_Tracking.md 'Pre-Registered Prediction'). It is not "
            "scoped to touch the test split at all, regardless of "
            "--confirm-final - PAD_UFES20's test split is also already "
            "twice-consumed and locked by test_split_guard.py."
        )

    if args.split == "test" and not args.confirm_final:
        raise SystemExit(
            "Refusing to evaluate on the test split without --confirm-final. "
            "The test split must stay untouched until the final Stage 1 "
            "evaluation run (see Project_Tracking.md decision 4)."
        )

    if args.branch == "cross_attention_backbone_ensemble":
        if args.seed is None:
            raise SystemExit("--branch cross_attention_backbone_ensemble requires --seed.")
        if args.ensemble_seeds is not None or args.tta:
            raise SystemExit(
                "--ensemble-seeds/--tta are not supported with "
                "--branch cross_attention_backbone_ensemble (unweighted, "
                "same-seed, two-backbone averaging only)."
            )
    else:
        if args.ensemble_seeds is not None and args.seed is not None:
            raise SystemExit("Pass either --seed or --ensemble-seeds, not both.")
        if args.ensemble_seeds is None and args.seed is None:
            raise SystemExit("One of --seed or --ensemble-seeds is required.")
        if args.ensemble_seeds is not None and args.branch != "cross_attention":
            raise SystemExit("--ensemble-seeds currently requires --branch cross_attention.")
        if args.tta and args.ensemble_seeds is None:
            raise SystemExit("--tta currently requires --ensemble-seeds too.")

    ds_config = get_dataset(args.dataset)

    if ds_config.image_branch_only and args.branch != "image":
        raise SystemExit(
            f"{args.dataset!r} is image_branch_only=True (see "
            f"Project_Tracking.md, 'Step 2 Integration Plan', 2026-07-29). "
            f"--branch {args.branch!r} is not valid for this dataset - use "
            f"--branch image, or --dataset PAD_UFES20 for metadata/fusion eval."
        )

    if args.branch == "cross_attention_backbone_ensemble":
        backbone_a = args.backbone_a or "convnext_tiny"
        backbone_b = args.backbone_b or "densenet121"
        result = evaluate_dual_backbone_ensemble(
            args.seed, args.split, backbone_a=backbone_a, backbone_b=backbone_b
        )
        reports_dir = ds_config.fusion_reports_dir.parent / "cross_attention_backbone"
        out_name = (
            f"eval_cross_attention_backbone_ensemble_{backbone_a}_{backbone_b}_"
            f"seed{args.seed}_{args.split}.json"
        )
    elif args.ensemble_seeds is not None:
        seeds = [int(s) for s in args.ensemble_seeds.split(",")]
        result = evaluate_ensemble(args.dataset, args.branch, seeds, args.split, tta=args.tta)
        reports_dir = ds_config.baseline_reports_dir.parent / "score_experiments"
        tta_suffix = "_tta" if args.tta else ""
        out_name = f"eval_{args.branch}_ensemble{len(seeds)}{tta_suffix}_{args.split}.json"
    else:
        result = evaluate(
            args.dataset, args.branch, args.seed, args.split,
            backbone=args.backbone, sampler=args.sampler, strong_augment=args.strong_augment,
            backbone_a=args.backbone_a, backbone_b=args.backbone_b,
        )
        if args.branch == "fusion":
            reports_dir = ds_config.fusion_reports_dir
        elif args.branch in ("cross_attention", "cross_attention_improved"):
            reports_dir = ds_config.baseline_reports_dir.parent / "score_experiments"
        elif args.branch == "backbone_fusion":
            reports_dir = ds_config.fusion_reports_dir.parent / "backbone_fusion"
        elif args.branch == "cross_attention_backbone":
            reports_dir = ds_config.fusion_reports_dir.parent / "cross_attention_backbone"
        elif args.branch == "cross_attention_efficientnet_expanded":
            reports_dir = ds_config.fusion_reports_dir.parent / "cross_attention_efficientnet_expanded"
        else:
            reports_dir = ds_config.baseline_reports_dir

        if args.branch == "image":
            run_name = _image_run_name(args.seed, args.backbone, args.sampler, args.strong_augment)
            out_name = f"eval_{run_name}_{args.split}.json"
        elif args.branch == "backbone_fusion":
            out_name = (
                f"eval_backbone_fusion_{args.backbone_a}_{args.backbone_b}_"
                f"seed{args.seed}_{args.split}.json"
            )
        elif args.branch == "cross_attention_backbone":
            out_name = (
                f"eval_cross_attention_backbone_{args.backbone}_seed{args.seed}_{args.split}.json"
            )
        else:
            out_name = f"eval_{args.branch}_seed{args.seed}_{args.split}.json"

    predictions_df = result.pop("_predictions_df", None)

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / out_name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if predictions_df is not None and args.split == "test":
        # Per-row test predictions, only written for the one-time final
        # evaluation - needed to pair-resample against the locked
        # cross_attention (EfficientNet-B0) test predictions in
        # bootstrap_significance_backbone_fusion.py.
        pred_out_name = "predictions_" + out_name[len("eval_"):].replace(".json", ".csv")
        pred_out_path = reports_dir / pred_out_name
        predictions_df.to_csv(pred_out_path, index=False)
        print(f"predictions written -> {pred_out_path}")

    print(f"macro-F1 ({args.split}): {result['macro_f1']:.4f}")
    print("per-class F1:")
    for name, score in result["per_class_f1"].items():
        print(f"  {name}: {score:.4f}")
    print(f"accuracy (reference only): {result['accuracy_reference_only']:.4f}")
    print(f"written -> {out_path}")


if __name__ == "__main__":
    main()
