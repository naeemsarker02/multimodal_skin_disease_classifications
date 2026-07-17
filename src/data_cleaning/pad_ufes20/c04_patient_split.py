"""Step 4: Patient-wise dataset split.

Splits patients (not images) into train/val/test so no patient's
images can appear in more than one split (avoids data leakage). The
audit found 179 patients with more than one distinct diagnosis, so a
naive random image-level split could leak identity/lesion information
across splits - splitting is done at the patient level instead.

Stratification is approximated by each patient's *dominant* diagnosis
(the most frequent disease_label among their images/lesions), then
patients within each dominant-diagnosis group are shuffled with a
fixed seed and assigned to train/val/test greedily by cumulative image
count, targeting the 70/15/15 ratio. This keeps class balance roughly
consistent across splits while still respecting patient-level grouping.
"""

import numpy as np
import pandas as pd

from src.data_audit.common.io_utils import save_csv
from src.data_cleaning.config import PAD_UFES20_PROCESSED_DIR, SPLIT_RATIOS, SPLIT_SEED


def _assign_split_within_group(patient_ids: list, image_counts: dict, ratios: dict) -> dict:
    """Greedily assign a shuffled list of patients to splits by cumulative
    image count, so image-level proportions approximate the target ratios."""
    total_images = sum(image_counts[p] for p in patient_ids)
    train_target = ratios["train"] * total_images
    val_target = (ratios["train"] + ratios["val"]) * total_images

    assignment = {}
    cumulative = 0
    for patient_id in patient_ids:
        cumulative += image_counts[patient_id]
        if cumulative <= train_target:
            assignment[patient_id] = "train"
        elif cumulative <= val_target:
            assignment[patient_id] = "val"
        else:
            assignment[patient_id] = "test"
    return assignment


def run(logger, df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SPLIT_SEED)

    dominant_label = (
        df.groupby("patient_id")["disease_label"]
        .agg(lambda labels: labels.value_counts().idxmax())
    )
    image_counts = df.groupby("patient_id").size().to_dict()

    split_assignment: dict[str, str] = {}
    for label, group in dominant_label.groupby(dominant_label):
        patient_ids = list(group.index)
        rng.shuffle(patient_ids)
        split_assignment.update(_assign_split_within_group(patient_ids, image_counts, SPLIT_RATIOS))

    assignment_df = pd.DataFrame(
        {
            "patient_id": list(split_assignment.keys()),
            "dominant_disease_label": [dominant_label[p] for p in split_assignment],
            "n_images": [image_counts[p] for p in split_assignment],
            "split": list(split_assignment.values()),
        }
    ).sort_values("patient_id").reset_index(drop=True)

    save_csv(assignment_df, PAD_UFES20_PROCESSED_DIR / "patient_split_assignment.csv")

    df = df.copy()
    df["split"] = df["patient_id"].map(split_assignment)

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].drop(columns=["split"])
        splits[split_name] = split_df
        save_csv(split_df, PAD_UFES20_PROCESSED_DIR / f"metadata_{split_name}.csv")
        logger.info(
            "%s: %d images, %d patients",
            split_name,
            len(split_df),
            split_df["patient_id"].nunique(),
        )

    logger.info("Split seed: %d | Target ratios: %s", SPLIT_SEED, SPLIT_RATIOS)
    logger.info("Saved -> %s", PAD_UFES20_PROCESSED_DIR / "patient_split_assignment.csv")
    for split_name in ["train", "val", "test"]:
        logger.info("Saved -> %s", PAD_UFES20_PROCESSED_DIR / f"metadata_{split_name}.csv")

    return splits
