"""Step 4: Lesion-wise dataset split.

HAM10000 has no patient identifier (audit section 11) - lesion_id is
the only grouping key, and 1,956 lesions have multiple images. Splits
lesions (not images) into train/val/test so no lesion's images can
appear in more than one split (avoids data leakage).

Stratification uses each lesion's dx label (the audit found 0 lesions
with inconsistent/multiple dx labels, so this is unambiguous - unlike
PAD-UFES-20's "dominant diagnosis" approximation). Lesions within each
label group are shuffled with a fixed seed and assigned to train/val/test
greedily by cumulative image count, targeting the 70/15/15 ratio.
"""

import numpy as np
import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import HAM10000_PROCESSED_DIR, SPLIT_RATIOS, SPLIT_SEED


def _assign_split_within_group(lesion_ids: list, image_counts: dict, ratios: dict) -> dict:
    """Greedily assign a shuffled list of lesions to splits by cumulative
    image count, so image-level proportions approximate the target ratios."""
    total_images = sum(image_counts[l] for l in lesion_ids)
    train_target = ratios["train"] * total_images
    val_target = (ratios["train"] + ratios["val"]) * total_images

    assignment = {}
    cumulative = 0
    for lesion_id in lesion_ids:
        cumulative += image_counts[lesion_id]
        if cumulative <= train_target:
            assignment[lesion_id] = "train"
        elif cumulative <= val_target:
            assignment[lesion_id] = "val"
        else:
            assignment[lesion_id] = "test"
    return assignment


def run(logger, df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SPLIT_SEED)

    lesion_label = df.groupby("lesion_id")["disease_label"].agg(lambda labels: labels.iloc[0])
    n_distinct = df.groupby("lesion_id")["disease_label"].nunique()
    inconsistent = n_distinct[n_distinct > 1]
    if not inconsistent.empty:
        logger.error("Lesions with inconsistent disease_label found: %s", list(inconsistent.index))
        raise ValueError("Lesion-level label inconsistency - cannot stratify by lesion label")

    image_counts = df.groupby("lesion_id").size().to_dict()

    split_assignment: dict[str, str] = {}
    for label, group in lesion_label.groupby(lesion_label):
        lesion_ids = list(group.index)
        rng.shuffle(lesion_ids)
        split_assignment.update(_assign_split_within_group(lesion_ids, image_counts, SPLIT_RATIOS))

    assignment_df = pd.DataFrame(
        {
            "lesion_id": list(split_assignment.keys()),
            "disease_label": [lesion_label[l] for l in split_assignment],
            "n_images": [image_counts[l] for l in split_assignment],
            "split": list(split_assignment.values()),
        }
    ).sort_values("lesion_id").reset_index(drop=True)

    save_csv(assignment_df, HAM10000_PROCESSED_DIR / "lesion_split_assignment.csv")

    df = df.copy()
    df["split"] = df["lesion_id"].map(split_assignment)

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].drop(columns=["split"])
        splits[split_name] = split_df
        save_csv(split_df, HAM10000_PROCESSED_DIR / f"metadata_{split_name}.csv")
        logger.info(
            "%s: %d images, %d lesions",
            split_name,
            len(split_df),
            split_df["lesion_id"].nunique(),
        )

    logger.info("Split seed: %d | Target ratios: %s", SPLIT_SEED, SPLIT_RATIOS)
    logger.info("Saved -> %s", HAM10000_PROCESSED_DIR / "lesion_split_assignment.csv")
    for split_name in ["train", "val", "test"]:
        logger.info("Saved -> %s", HAM10000_PROCESSED_DIR / f"metadata_{split_name}.csv")

    return splits
