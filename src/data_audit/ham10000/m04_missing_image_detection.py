"""Module 5: Missing image detection.

Cross-references metadata.csv's image_id column against the image
inventory in both directions. Unlike PAD-UFES-20's img_id (which
already includes the file extension), HAM10000's image_id is the bare
ISIC identifier (e.g. "ISIC_0027419") with no extension, so the
expected filename is reconstructed as f"{image_id}.jpg" before
comparison.
"""

import pandas as pd

from src.data_audit.config import HAM10000_METADATA_FILE, HAM10000_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger, inventory_df: pd.DataFrame) -> dict:
    metadata = pd.read_csv(HAM10000_METADATA_FILE)
    metadata["expected_filename"] = metadata["image_id"] + ".jpg"

    expected_filenames = set(metadata["expected_filename"])
    inventory_filenames = set(inventory_df["filename"])

    missing_filenames = sorted(expected_filenames - inventory_filenames)
    orphan_files = sorted(inventory_filenames - expected_filenames)

    missing_df = metadata[metadata["expected_filename"].isin(missing_filenames)][
        ["lesion_id", "image_id", "expected_filename"]
    ]
    orphan_df = inventory_df[inventory_df["filename"].isin(orphan_files)][
        ["source_folder", "filename", "relative_path"]
    ]

    duplicate_image_id = metadata[metadata.duplicated("image_id", keep=False)][
        ["lesion_id", "image_id"]
    ].sort_values("image_id")

    save_csv(missing_df, HAM10000_REPORTS_DIR / "04_missing_images.csv")
    save_csv(orphan_df, HAM10000_REPORTS_DIR / "04_orphan_images.csv")
    save_csv(duplicate_image_id, HAM10000_REPORTS_DIR / "04_duplicate_image_id.csv")

    logger.info("Metadata rows: %d | Inventory files: %d", len(metadata), len(inventory_df))
    logger.info("Missing images (in metadata, not on disk): %d", len(missing_df))
    logger.info("Orphan images (on disk, not in metadata): %d", len(orphan_df))
    logger.info("Duplicate image_id entries in metadata: %d", len(duplicate_image_id))

    return {
        "metadata_rows": len(metadata),
        "inventory_files": len(inventory_df),
        "missing_count": len(missing_df),
        "orphan_count": len(orphan_df),
        "duplicate_image_id_count": len(duplicate_image_id),
    }
