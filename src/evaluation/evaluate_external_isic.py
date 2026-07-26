"""Phase 8 - HAM10000 -> ISIC external validation (scope approved
2026-07-25, Project_Tracking.md "Proposed ISIC External Validation
Scope").

Usage:
    python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_1 --variant image --seed 0
    python -m src.evaluation.evaluate_external_isic --archive ISIC_Archive_2 --variant metadata --seed 0

Zero-shot transfer, same protocol as evaluate_cross_dataset.py's PAD->HAM
experiment (Protocol A): a HAM10000-trained model's native, UNMODIFIED
classifier (full argmax over its own 7-class output, never masked) is run
on the target ISIC archive; scoring is restricted to the classes that
exist in both taxonomies (exact-string match only - AK and the
keratosis-family naming/granularity mismatches are deliberately NOT
merged, per the 2026-07-25 scope decision - conservative, no assumed
clinical equivalence, consistent with Protocol A).

Archive 1 has 0 usable metadata columns (image-only by necessity) - only
the image branch runs there. Archive 2 has usable metadata but its
anatom_site_general vocabulary needs mapping into HAM10000's
anatomical_site vocabulary first - see
docs/Phase8_ISIC_Archive2_Anatomical_Site_Mapping.csv (approved
2026-07-25) and
config.normalize_isic_archive2_anatomical_site_for_ham10000().

Exclusions applied per archive, both required before scoring:
    data/processed/<archive>/external_validation_exclusions.csv
        (2026-07-08 - drops images already seen by the HAM10000-trained
        model, i.e. overlapping with HAM10000 itself)
    data/processed/<archive>/label_conflict_exclusions.csv
        (2026-07-25 - drops the 3 images with disagreeing ground truth
        between ISIC Archive 1 and ISIC Archive 2)

Not gated by test_split_guard.py: this evaluates HAM10000's already-
finalized *training* checkpoints against ISIC's own data - it is not a
second read of HAM10000's own test split, and neither ISIC archive's own
train/val/test split was ever used for ISIC-internal model training in
this project (no ISIC-trained models exist).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from src.models.config import (
    BATCH_SIZE,
    get_dataset,
    normalize_isic_archive2_anatomical_site_for_ham10000,
    resolve_image_path,
)
from src.models.dataset import MetadataPreprocessor, build_image_transform
from src.models.image_model import build_efficientnet_b0
from src.models.metadata_model import MetadataMLP

ARCHIVE_SHARED_CLASSES = {
    # Exact-string-match only with HAM10000's taxonomy (Protocol A
    # precedent) - AK and keratosis-family naming/granularity mismatches
    # deliberately deferred, not merged (2026-07-25 scope decision).
    "ISIC_Archive_1": ["Basal Cell Carcinoma", "Dermatofibroma", "Melanoma", "Nevus", "Vascular Lesion"],
    "ISIC_Archive_2": ["Basal Cell Carcinoma", "Dermatofibroma", "Melanoma", "Nevus"],
}

ARCHIVE_HAS_METADATA = {
    "ISIC_Archive_1": False,
    "ISIC_Archive_2": True,
}

ARCHIVE_METADATA_COLUMNS = {
    # archive column name -> HAM10000-schema column name. anatom_site_general
    # additionally passes through the ISIC Archive 2 -> HAM10000 mapping
    # (see normalize_isic_archive2_anatomical_site_for_ham10000).
    "ISIC_Archive_2": {"age_approx": "age", "sex": "sex", "anatom_site_general": "anatomical_site"},
}


def _archive_processed_dir(archive: str) -> Path:
    return Path("data") / "processed" / archive


def load_archive_metadata(archive: str) -> pd.DataFrame:
    processed_dir = _archive_processed_dir(archive)
    dfs = [pd.read_csv(processed_dir / f"metadata_{split}.csv") for split in ("train", "val", "test")]
    return pd.concat(dfs, ignore_index=True)


def apply_exclusions(df: pd.DataFrame, archive: str) -> pd.DataFrame:
    processed_dir = _archive_processed_dir(archive)
    ham_overlap = set(pd.read_csv(processed_dir / "external_validation_exclusions.csv")["image_id"])
    label_conflicts = set(pd.read_csv(processed_dir / "label_conflict_exclusions.csv")["image_id"])
    excluded = ham_overlap | label_conflicts
    return df[~df["image_id"].isin(excluded)].reset_index(drop=True)


class ExternalIsicEvalDataset(Dataset):
    """ISIC archive rows, filtered to that archive's shared-class list and
    with both exclusion lists applied. Labels are encoded in HAM10000's
    label_to_idx space (the model's own output space) - the shared class
    name strings are identical across HAM10000 and both ISIC archives'
    disease_label vocabularies (verified during dataset preparation, same
    harmonized taxonomy used project-wide).
    """

    def __init__(self, archive: str, ham_label_to_idx: dict, need_metadata: bool,
                 eval_preprocessor: MetadataPreprocessor = None):
        df = load_archive_metadata(archive)
        df = apply_exclusions(df, archive)
        shared_classes = ARCHIVE_SHARED_CLASSES[archive]
        self.df = df[df["disease_label"].isin(shared_classes)].reset_index(drop=True)
        self.label_to_idx = ham_label_to_idx
        self.transform = build_image_transform(train=False)
        self.need_metadata = need_metadata
        self.preprocessor = eval_preprocessor
        self.archive = archive
        if need_metadata and eval_preprocessor is None:
            raise ValueError("eval_preprocessor required when need_metadata=True")

    def __len__(self) -> int:
        return len(self.df)

    def _adapted_row(self, row: pd.Series) -> pd.Series:
        """Archive-native column names/vocab -> HAM10000-schema row, for
        MetadataPreprocessor.transform_row(). Archive 1 never reaches here
        (no metadata branch); only Archive 2 is handled.
        """
        col_map = ARCHIVE_METADATA_COLUMNS[self.archive]
        adapted = {}
        for archive_col, ham_col in col_map.items():
            value = row[archive_col]
            if ham_col == "anatomical_site":
                value = normalize_isic_archive2_anatomical_site_for_ham10000(value)
            adapted[ham_col] = value
        return pd.Series(adapted)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self.label_to_idx[row["disease_label"]]
        if self.need_metadata:
            metadata = self.preprocessor.transform_row(self._adapted_row(row))
            return image, metadata, label
        return image, label


def build_ham_eval_preprocessor(ham_ds_config) -> MetadataPreprocessor:
    """Fits on HAM10000's own train split, in HAM10000's own schema
    (age/sex/anatomical_site) - no column_transforms needed here, since
    the ISIC Archive 2 -> HAM10000 normalization is applied to Archive
    2's raw values in ExternalIsicEvalDataset._adapted_row() before this
    preprocessor ever sees them, not via a transform hook on the fit
    side (unlike the PAD->HAM10000 experiment, HAM10000 and Archive 2
    don't share raw column names, so there's no single df both could be
    fit/transformed from).
    """
    train_df = pd.read_csv(ham_ds_config.train_csv)
    return MetadataPreprocessor(ham_ds_config).fit(train_df)


def load_model_for_variant(variant: str, seed: int, ham_ds_config, num_classes: int,
                            metadata_input_dim: int, device) -> torch.nn.Module:
    ckpt_path = ham_ds_config.checkpoints_dir / f"{variant}_seed{seed}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if variant == "image":
        model = build_efficientnet_b0(num_classes=num_classes)
    elif variant == "metadata":
        model = MetadataMLP(input_dim=metadata_input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate(archive: str, variant: str, seed: int) -> dict:
    if variant == "metadata" and not ARCHIVE_HAS_METADATA[archive]:
        raise ValueError(f"{archive} has no usable metadata columns - only --variant image is valid here.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ham_ds_config = get_dataset("HAM10000")

    need_metadata = variant == "metadata"
    eval_preprocessor = None
    metadata_input_dim = None
    if need_metadata:
        eval_preprocessor = build_ham_eval_preprocessor(ham_ds_config)
        metadata_input_dim = eval_preprocessor.output_dim

    dataset = ExternalIsicEvalDataset(archive, ham_ds_config.label_to_idx, need_metadata, eval_preprocessor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model_for_variant(variant, seed, ham_ds_config, ham_ds_config.num_classes, metadata_input_dim, device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if need_metadata:
                _, metadata, labels = batch
                outputs = model(metadata.to(device))
            else:
                images, labels = batch
                outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    class_names = ham_ds_config.class_names
    shared_classes = ARCHIVE_SHARED_CLASSES[archive]
    shared_indices = [ham_ds_config.label_to_idx[c] for c in shared_classes]

    macro_f1 = f1_score(all_labels, all_preds, labels=shared_indices, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, labels=shared_indices, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    spillover_count = sum(1 for p in all_preds if p not in shared_indices)

    result = {
        "experiment": "phase8_ham_to_isic_external_validation",
        "protocol": "A - full native argmax, restricted to exact-string-match shared classes for scoring",
        "archive": archive,
        "variant": variant,
        "seed": seed,
        "n_eval_rows": len(dataset),
        "shared_classes": shared_classes,
        "macro_f1_shared_classes": macro_f1,
        "per_class_f1_shared_classes": dict(zip(shared_classes, per_class_f1.tolist())),
        "spillover_count": spillover_count,
        "spillover_rate": spillover_count / len(all_labels) if all_labels else 0.0,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_class_order": class_names,
    }

    predictions_df = dataset.df[["image_id", "image_path"]].copy()
    predictions_df["true_label_idx"] = all_labels
    predictions_df["true_label_name"] = [class_names[i] for i in all_labels]
    predictions_df["pred_label_idx"] = all_preds
    predictions_df["pred_label_name"] = [class_names[i] for i in all_preds]

    return result, predictions_df


def main():
    parser = argparse.ArgumentParser(description="Phase 8: HAM10000 -> ISIC external validation")
    parser.add_argument("--archive", choices=["ISIC_Archive_1", "ISIC_Archive_2"], required=True)
    parser.add_argument("--variant", choices=["image", "metadata"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    result, predictions_df = evaluate(args.archive, args.variant, args.seed)

    reports_dir = Path("reports") / "HAM10000" / "external_isic"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"eval_{args.archive}_{args.variant}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    predictions_path = reports_dir / f"predictions_{args.archive}_{args.variant}_seed{args.seed}.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print(f"n_eval_rows: {result['n_eval_rows']}")
    print(f"macro-F1 (shared classes): {result['macro_f1_shared_classes']:.4f}")
    print("per-class F1 (shared classes):")
    for name, score in result["per_class_f1_shared_classes"].items():
        print(f"  {name}: {score:.4f}")
    print(f"spillover rate (predicted a non-shared HAM10000 class): {result['spillover_rate']:.4f}")
    print(f"written -> {out_path}")
    print(f"written -> {predictions_path}")


if __name__ == "__main__":
    main()
