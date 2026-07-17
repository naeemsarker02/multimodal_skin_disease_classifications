"""Module 5: Missing image detection.

Cross-references metadata.csv's img_id column against the image
inventory in both directions:
  - metadata rows whose img_id has no corresponding file on disk (missing)
  - files on disk not referenced by any img_id in metadata (orphans)

Also flags duplicate img_id values in metadata and duplicate filenames
across image folders, since both fall out of the same set-reconciliation
logic and are direct threats to a clean 1:1 image<->metadata mapping.
"""

import pandas as pd

from src.data_audit.config import PAD_UFES20_METADATA_FILE, PAD_UFES20_REPORTS_DIR
from src.data_audit.common.io_utils import save_csv


def run(logger, inventory_df: pd.DataFrame) -> dict:
    metadata = pd.read_csv(PAD_UFES20_METADATA_FILE)

    metadata_ids = set(metadata["img_id"])
    inventory_filenames = set(inventory_df["filename"])

    missing_ids = sorted(metadata_ids - inventory_filenames)
    orphan_files = sorted(inventory_filenames - metadata_ids)

    missing_df = metadata[metadata["img_id"].isin(missing_ids)][
        ["patient_id", "lesion_id", "img_id"]
    ]
    orphan_df = inventory_df[inventory_df["filename"].isin(orphan_files)][
        ["source_folder", "filename", "relative_path"]
    ]

    duplicate_img_id = metadata[metadata.duplicated("img_id", keep=False)][
        ["patient_id", "lesion_id", "img_id"]
    ].sort_values("img_id")

    save_csv(missing_df, PAD_UFES20_REPORTS_DIR / "04_missing_images.csv")
    save_csv(orphan_df, PAD_UFES20_REPORTS_DIR / "04_orphan_images.csv")
    save_csv(duplicate_img_id, PAD_UFES20_REPORTS_DIR / "04_duplicate_img_id.csv")

    logger.info("Metadata rows: %d | Inventory files: %d", len(metadata), len(inventory_df))
    logger.info("Missing images (in metadata, not on disk): %d", len(missing_df))
    logger.info("Orphan images (on disk, not in metadata): %d", len(orphan_df))
    logger.info("Duplicate img_id entries in metadata: %d", len(duplicate_img_id))

    return {
        "metadata_rows": len(metadata),
        "inventory_files": len(inventory_df),
        "missing_count": len(missing_df),
        "orphan_count": len(orphan_df),
        "duplicate_img_id_count": len(duplicate_img_id),
    }
