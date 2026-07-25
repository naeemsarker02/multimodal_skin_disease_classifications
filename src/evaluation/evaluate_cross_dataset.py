"""Phase 8 Experiment 1 - PAD-UFES-20 -> HAM10000 cross-dataset
generalization, Protocol A (approved 2026-07-18, Project_Tracking.md).

Usage:
    python -m src.evaluation.evaluate_cross_dataset --variant image --seed 0
    python -m src.evaluation.evaluate_cross_dataset --variant cross_attention --seed 0

Zero-shot transfer: a model trained only on PAD-UFES-20 (never sees
HAM10000 during training) is evaluated directly on HAM10000's test split
- the model's weights never change here, only the evaluation data does.

Protocol A:
- Filter HAM10000's test split to the 3 classes that exist in both
  datasets' taxonomies (Basal Cell Carcinoma, Melanoma, Nevus - PAD-UFES-20's
  other 3 classes and HAM10000's other 4 have no counterpart in the other
  dataset). 1,253 of HAM10000's 1,510 test rows fall in these 3 classes.
- Run each PAD-UFES-20-trained model's native, UNMODIFIED classifier (full
  argmax over its own output classes - never masked/restricted to just
  the 3 shared classes) on those images. A prediction of one of PAD-UFES-20's
  3 non-transferable classes on a true-shared-class HAM10000 image simply
  counts as wrong - this preserves real difficulty rather than artificially
  inflating transfer performance via masking.
- Report macro-F1 AND per-class F1 restricted to the 3 shared classes
  (sklearn's `labels=` parameter - precision/recall for each shared class
  still accounts for every prediction, including "spillover" into a
  non-shared class, which only shows up as a false negative for the true
  class), plus the full confusion matrix (over all 6 PAD-UFES-20 classes)
  so the spillover pattern itself is visible, not hidden.

Metadata schema: the metadata/late_fusion/cross_attention variants use
the *reduced-feature* PAD-UFES-20 checkpoints (metadata_reduced_seed{N},
fusion_reduced_seed{N}, cross_attention_reduced_seed{N} - see
train_metadata_reduced.py et al.), restricted to the 3 columns HAM10000
also has, with anatomical_site normalized per
docs/Phase8_Anatomical_Site_Mapping.csv. HAM10000's own sex/anatomical_site
values are already in the target vocabulary (no transform applied to
them - see MetadataPreprocessor.without_transforms()).

HAM10000's test split has never been used for any tuning decision
(Stage 1 used only train/val) - this generalization experiment is exactly
the "single final evaluation run" the project's val/test discipline was
designed for.
"""

import argparse
import json

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from src.models.config import (
    BATCH_SIZE,
    REDUCED_CATEGORICAL_FEATURES,
    REDUCED_NUMERIC_FEATURES,
    get_dataset,
    normalize_anatomical_site_for_cross_dataset,
    resolve_image_path,
)
from src.models.cross_attention_fusion_model import CrossAttentionFusionModel
from src.models.dataset import MetadataPreprocessor, build_image_transform
from src.models.fusion_model import FusionModel
from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP

SHARED_CLASSES = ["Basal Cell Carcinoma", "Melanoma", "Nevus"]

VARIANT_NEEDS_METADATA = {
    "image": False,
    "metadata": True,
    "late_fusion": True,
    "cross_attention": True,
}
VARIANT_CHECKPOINT_NAME = {
    "image": "image_seed{seed}_best.pt",
    "metadata": "metadata_reduced_seed{seed}_best.pt",
    "late_fusion": "fusion_reduced_seed{seed}_best.pt",
    "cross_attention": "cross_attention_reduced_seed{seed}_best.pt",
}


class CrossDatasetEvalDataset(Dataset):
    """HAM10000 test rows restricted to SHARED_CLASSES. Labels are encoded
    in PAD-UFES-20's label_to_idx space (the model's own output space),
    not HAM10000's - the 3 shared class name strings are identical across
    both datasets' label_mapping.csv (verified during dataset preparation).
    """

    def __init__(self, ham_test_csv, pad_label_to_idx, need_metadata: bool,
                 eval_preprocessor: MetadataPreprocessor = None):
        df = pd.read_csv(ham_test_csv)
        self.df = df[df["disease_label"].isin(SHARED_CLASSES)].reset_index(drop=True)
        self.label_to_idx = pad_label_to_idx
        self.transform = build_image_transform(train=False)
        self.need_metadata = need_metadata
        self.preprocessor = eval_preprocessor
        if need_metadata and eval_preprocessor is None:
            raise ValueError("eval_preprocessor required when need_metadata=True")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.label_to_idx[row["disease_label"]]
        if self.need_metadata:
            metadata = self.preprocessor.transform_row(row)
            return image, metadata, label
        return image, label


def build_pad_to_ham_eval_preprocessor(pad_ds_config):
    """Fits on PAD-UFES-20's train split (age, sex, anatomical_site only,
    anatomical_site normalized into HAM10000's vocabulary), then returns
    an eval-mode copy with the normalization transform cleared - HAM10000's
    own values are already in the target vocabulary and pass straight
    through the default identity path.
    """
    reduced_config = pad_ds_config.with_features(REDUCED_NUMERIC_FEATURES, REDUCED_CATEGORICAL_FEATURES)
    train_df = pd.read_csv(reduced_config.train_csv)
    fit_preprocessor = MetadataPreprocessor(
        reduced_config,
        column_transforms={"anatomical_site": normalize_anatomical_site_for_cross_dataset},
    ).fit(train_df)
    return reduced_config, fit_preprocessor.without_transforms()


def coverage_diagnostic(dataset: CrossDatasetEvalDataset) -> dict:
    """% of HAM10000 rows whose sex/anatomical_site value matched a known
    PAD-UFES-20-fitted category, vs. fell to the "__MISSING__" bucket -
    reported so any residual vocabulary mismatch is visible, not silently
    absorbed. Some genuine fall-through is expected and correct (e.g.
    HAM10000's "genital"/"trunk" categories have no PAD-UFES-20 site that
    maps to them).
    """
    if dataset.preprocessor is None:
        return {}
    result = {}
    for col in ("sex", "anatomical_site"):
        categories = set(dataset.preprocessor.categorical_values[col])
        values = dataset.df[col].apply(lambda v: dataset.preprocessor._categorical_value(col, v))
        matched = values.isin(categories - {"__MISSING__"}).sum()
        result[col] = {
            "matched": int(matched),
            "fell_to_missing": int(len(values) - matched),
            "total": int(len(values)),
        }
    return result


def load_model_for_variant(variant: str, seed: int, pad_ds_config, num_classes: int,
                            metadata_input_dim: int, device) -> torch.nn.Module:
    if variant == "image":
        model = build_efficientnet_b0(num_classes=num_classes)
        ckpt_path = pad_ds_config.stage1_checkpoints_dir / VARIANT_CHECKPOINT_NAME[variant].format(seed=seed)
    elif variant == "metadata":
        model = MetadataMLP(input_dim=metadata_input_dim, num_classes=num_classes)
        ckpt_path = pad_ds_config.checkpoints_dir / VARIANT_CHECKPOINT_NAME[variant].format(seed=seed)
    elif variant == "late_fusion":
        model = FusionModel(metadata_input_dim=metadata_input_dim, num_classes=num_classes)
        ckpt_path = pad_ds_config.checkpoints_dir / VARIANT_CHECKPOINT_NAME[variant].format(seed=seed)
    elif variant == "cross_attention":
        model = CrossAttentionFusionModel(metadata_input_dim=metadata_input_dim, num_classes=num_classes)
        ckpt_path = pad_ds_config.checkpoints_dir / VARIANT_CHECKPOINT_NAME[variant].format(seed=seed)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(variant: str, seed: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_ds_config = get_dataset("PAD_UFES20")
    ham_ds_config = get_dataset("HAM10000")

    need_metadata = VARIANT_NEEDS_METADATA[variant]
    eval_preprocessor = None
    metadata_input_dim = None
    if need_metadata:
        _, eval_preprocessor = build_pad_to_ham_eval_preprocessor(pad_ds_config)
        metadata_input_dim = eval_preprocessor.output_dim

    dataset = CrossDatasetEvalDataset(
        ham_ds_config.test_csv, pad_ds_config.label_to_idx, need_metadata, eval_preprocessor
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model_for_variant(
        variant, seed, pad_ds_config, pad_ds_config.num_classes, metadata_input_dim, device
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if need_metadata:
                images, metadata, labels = batch
                if variant == "metadata":
                    outputs = model(metadata.to(device))
                else:
                    outputs = model(images.to(device), metadata.to(device))
            else:
                images, labels = batch
                outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    class_names = pad_ds_config.class_names
    shared_indices = [pad_ds_config.label_to_idx[c] for c in SHARED_CLASSES]

    macro_f1 = f1_score(all_labels, all_preds, labels=shared_indices, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, labels=shared_indices, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    spillover_count = sum(
        1 for p, l in zip(all_preds, all_labels) if p not in shared_indices
    )

    result = {
        "experiment": "phase8_pad_to_ham_cross_dataset_generalization",
        "protocol": "A - full native argmax, restricted to 3 shared classes for scoring",
        "variant": variant,
        "seed": seed,
        "n_eval_rows": len(dataset),
        "shared_classes": SHARED_CLASSES,
        "macro_f1_shared_classes": macro_f1,
        "per_class_f1_shared_classes": dict(zip(SHARED_CLASSES, per_class_f1.tolist())),
        "spillover_count": spillover_count,
        "spillover_rate": spillover_count / len(all_labels),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_class_order": class_names,
        "metadata_coverage": coverage_diagnostic(dataset) if need_metadata else {},
    }

    # Row-level predictions, ordered identically to `dataset.df` (loader uses
    # shuffle=False) - saved so downstream analyses (e.g. bootstrap
    # significance testing) can resample individual rows without re-running
    # inference. Not part of the aggregate metrics above; purely additional.
    predictions_df = dataset.df[["image_path"]].copy()
    predictions_df["true_label_idx"] = all_labels
    predictions_df["true_label_name"] = [class_names[i] for i in all_labels]
    predictions_df["pred_label_idx"] = all_preds
    predictions_df["pred_label_name"] = [class_names[i] for i in all_preds]

    return result, predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 Experiment 1: PAD-UFES-20 -> HAM10000 cross-dataset generalization"
    )
    parser.add_argument(
        "--variant", choices=["image", "metadata", "late_fusion", "cross_attention"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    result, predictions_df = evaluate(args.variant, args.seed)

    ds_config = get_dataset("PAD_UFES20")
    reports_dir = ds_config.baseline_reports_dir.parent / "cross_dataset"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"eval_{args.variant}_seed{args.seed}_pad_to_ham.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    predictions_path = reports_dir / f"predictions_{args.variant}_seed{args.seed}_pad_to_ham.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print(f"macro-F1 (3 shared classes): {result['macro_f1_shared_classes']:.4f}")
    print("per-class F1 (shared classes):")
    for name, score in result["per_class_f1_shared_classes"].items():
        print(f"  {name}: {score:.4f}")
    print(f"spillover rate (predicted a non-shared PAD-UFES-20 class): {result['spillover_rate']:.4f}")
    if result["metadata_coverage"]:
        print("metadata coverage:", result["metadata_coverage"])
    print(f"written -> {out_path}")
    print(f"written -> {predictions_path}")


if __name__ == "__main__":
    main()
