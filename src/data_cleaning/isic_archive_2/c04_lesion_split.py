"""Step 4: Lesion-wise dataset split.

lesion_id is populated for 23,664/25,331 rows (93.4%); patient_id for
only 417 (audit section 8). Rows without a lesion_id are treated as
their own singleton group keyed by image_id, so no image is silently
dropped from the split, and no group spans more than one split.

Rows with no disease_label (missing diagnosis_3) are excluded here -
they cannot be stratified or used for classification.

Stratification uses each lesion's disease_label (the audit confirmed 0
lesions have inconsistent diagnosis_3 across their images). Groups
within each label are shuffled with a fixed seed and assigned to
train/val/test greedily by cumulative image count.
"""

import numpy as np
import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import ISIC2_PROCESSED_DIR, SPLIT_RATIOS, SPLIT_SEED


def _assign_split_within_group(group_ids: list, image_counts: dict, ratios: dict) -> dict:
    total_images = sum(image_counts[g] for g in group_ids)
    train_target = ratios["train"] * total_images
    val_target = (ratios["train"] + ratios["val"]) * total_images

    assignment = {}
    cumulative = 0
    for group_id in group_ids:
        cumulative += image_counts[group_id]
        if cumulative <= train_target:
            assignment[group_id] = "train"
        elif cumulative <= val_target:
            assignment[group_id] = "val"
        else:
            assignment[group_id] = "test"
    return assignment


def run(logger, df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SPLIT_SEED)

    labeled_df = df[df["disease_label"].notna()].copy()
    logger.info("Rows with a disease_label available for splitting: %d/%d", len(labeled_df), len(df))

    labeled_df["group_id"] = labeled_df["lesion_id"].fillna(labeled_df["image_id"])

    n_distinct = labeled_df.groupby("group_id")["disease_label"].nunique()
    inconsistent = n_distinct[n_distinct > 1]
    if not inconsistent.empty:
        logger.error("Groups with inconsistent disease_label found: %s", list(inconsistent.index))
        raise ValueError("Group-level label inconsistency - cannot stratify by group label")

    group_label = labeled_df.groupby("group_id")["disease_label"].first()
    image_counts = labeled_df.groupby("group_id").size().to_dict()

    split_assignment: dict[str, str] = {}
    for label, group in group_label.groupby(group_label):
        group_ids = list(group.index)
        rng.shuffle(group_ids)
        split_assignment.update(_assign_split_within_group(group_ids, image_counts, SPLIT_RATIOS))

    assignment_df = pd.DataFrame(
        {
            "group_id": list(split_assignment.keys()),
            "disease_label": [group_label[g] for g in split_assignment],
            "n_images": [image_counts[g] for g in split_assignment],
            "split": list(split_assignment.values()),
        }
    ).sort_values("group_id").reset_index(drop=True)
    save_csv(assignment_df, ISIC2_PROCESSED_DIR / "group_split_assignment.csv")

    labeled_df["split"] = labeled_df["group_id"].map(split_assignment)

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_df = labeled_df[labeled_df["split"] == split_name].drop(columns=["split", "group_id"])
        splits[split_name] = split_df
        save_csv(split_df, ISIC2_PROCESSED_DIR / f"metadata_{split_name}.csv")
        logger.info("%s: %d images, %d groups", split_name, len(split_df), split_df.apply(lambda r: r["lesion_id"] if pd.notna(r["lesion_id"]) else r["image_id"], axis=1).nunique())

    logger.info("Split seed: %d | Target ratios: %s", SPLIT_SEED, SPLIT_RATIOS)
    logger.info("Saved -> %s", ISIC2_PROCESSED_DIR / "group_split_assignment.csv")

    return splits
