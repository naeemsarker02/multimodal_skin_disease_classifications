"""Module 3: Missing / orphan image detection.

Cross-references metadata.csv's isic_id column against the image
inventory in both directions, and flags duplicate isic_id rows.
"""

import pandas as pd

from src.data_audit.config import ISIC2_METADATA_FILE, ISIC2_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger, inventory_df: pd.DataFrame) -> dict:
    metadata = pd.read_csv(ISIC2_METADATA_FILE)
    metadata["expected_filename"] = metadata["isic_id"] + ".jpg"

    expected_filenames = set(metadata["expected_filename"])
    inventory_filenames = set(inventory_df["filename"])

    missing_filenames = sorted(expected_filenames - inventory_filenames)
    orphan_files = sorted(inventory_filenames - expected_filenames)

    missing_df = metadata[metadata["expected_filename"].isin(missing_filenames)][["isic_id", "expected_filename"]]
    orphan_df = inventory_df[inventory_df["filename"].isin(orphan_files)][["source_folder", "filename", "relative_path"]]
    duplicate_isic_id = metadata[metadata.duplicated("isic_id", keep=False)][["isic_id"]].sort_values("isic_id")

    save_csv(missing_df, ISIC2_REPORTS_DIR / "05_missing_images.csv")
    save_csv(orphan_df, ISIC2_REPORTS_DIR / "05_orphan_images.csv")
    save_csv(duplicate_isic_id, ISIC2_REPORTS_DIR / "05_duplicate_isic_id.csv")

    logger.info("Metadata rows: %d | Inventory files: %d", len(metadata), len(inventory_df))
    logger.info("Missing images (in metadata, not on disk): %d", len(missing_df))
    logger.info("Orphan images (on disk, not in metadata): %d", len(orphan_df))
    logger.info("Duplicate isic_id entries in metadata: %d", len(duplicate_isic_id))

    return {
        "metadata_rows": len(metadata),
        "inventory_files": len(inventory_df),
        "missing_count": len(missing_df),
        "orphan_count": len(orphan_df),
        "duplicate_isic_id_count": len(duplicate_isic_id),
    }
