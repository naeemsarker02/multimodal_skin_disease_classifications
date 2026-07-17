"""Module 11: Lesion statistics.

HAM10000 has no patient identifier - only lesion_id, and a lesion can
have multiple images (2-6) taken at different angles/times. This is
the HAM10000 equivalent of PAD-UFES-20's patient statistics module:
here the unit that must not be split across train/val/test is the
*lesion*, not the patient, since that is the only grouping key this
dataset provides.
"""

import pandas as pd

from src.data_audit.config import HAM10000_METADATA_FILE, HAM10000_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger) -> dict:
    metadata = pd.read_csv(HAM10000_METADATA_FILE)

    n_unique_lesions = metadata["lesion_id"].nunique()
    images_per_lesion = metadata.groupby("lesion_id").size()

    summary_df = pd.DataFrame(
        [
            {"metric": "n_unique_lesions", "value": n_unique_lesions},
            {"metric": "n_total_images", "value": len(metadata)},
            {"metric": "images_per_lesion_min", "value": images_per_lesion.min()},
            {"metric": "images_per_lesion_max", "value": images_per_lesion.max()},
            {"metric": "images_per_lesion_mean", "value": round(images_per_lesion.mean(), 2)},
            {"metric": "images_per_lesion_median", "value": images_per_lesion.median()},
            {"metric": "lesions_with_multiple_images", "value": int((images_per_lesion > 1).sum())},
        ]
    )
    save_csv(summary_df, HAM10000_REPORTS_DIR / "10_lesion_statistics.csv")

    # Integrity check: a lesion's images should all carry the same
    # diagnosis. If not, that is a data-quality issue worth flagging
    # before any lesion-wise split is built.
    diagnoses_per_lesion = metadata.groupby("lesion_id")["dx"].nunique()
    multi_dx_lesions = diagnoses_per_lesion[diagnoses_per_lesion > 1]
    multi_dx_df = metadata[metadata["lesion_id"].isin(multi_dx_lesions.index)][
        ["lesion_id", "image_id", "dx"]
    ].sort_values("lesion_id")
    save_csv(multi_dx_df, HAM10000_REPORTS_DIR / "10_lesions_multiple_diagnoses.csv")

    logger.info("Unique lesions: %d | Total images: %d", n_unique_lesions, len(metadata))
    logger.info(
        "Images per lesion: min=%d max=%d mean=%.2f median=%.1f",
        images_per_lesion.min(), images_per_lesion.max(),
        images_per_lesion.mean(), images_per_lesion.median(),
    )
    logger.info("Lesions with multiple images: %d", int((images_per_lesion > 1).sum()))
    logger.info("Lesions with inconsistent (multiple) dx labels: %d", len(multi_dx_lesions))
    logger.info("Saved -> %s", HAM10000_REPORTS_DIR / "10_lesion_statistics.csv")

    return {
        "n_unique_lesions": n_unique_lesions,
        "lesions_with_multiple_images": int((images_per_lesion > 1).sum()),
        "lesions_with_multiple_diagnoses": len(multi_dx_lesions),
    }
