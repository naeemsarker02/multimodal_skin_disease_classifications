"""Module 1: Folder structure + image inventory.

ISIC Archive 2 ships as a single flat `images/` folder plus one rich
metadata.csv (no pre-split, no per-class subfolders - unlike Archive 1).
"""

import pandas as pd

from src.data_audit.config import ISIC2_IMAGE_SUBDIRS, ISIC2_RAW_DIR, ISIC2_REPORTS_DIR, VALID_IMAGE_EXTENSIONS
from src.data_audit.common.io_utils import list_files, save_csv


def run(logger) -> pd.DataFrame:
    logger.info("Building folder structure + image inventory from %s", ISIC2_RAW_DIR)

    rows = []
    folder_rows = []
    for subdir in ISIC2_IMAGE_SUBDIRS:
        folder = ISIC2_RAW_DIR / subdir
        files = list_files(folder, VALID_IMAGE_EXTENSIONS)
        folder_rows.append({"folder": subdir, "num_files": len(files)})
        logger.info("  %s -> %d image files", subdir, len(files))
        for fpath in files:
            rows.append(
                {
                    "source_folder": subdir,
                    "filename": fpath.name,
                    "extension": fpath.suffix.lower(),
                    "size_bytes": fpath.stat().st_size,
                    "relative_path": str(fpath.relative_to(ISIC2_RAW_DIR)),
                }
            )

    inventory_df = pd.DataFrame(rows)
    folder_df = pd.DataFrame(folder_rows)

    save_csv(folder_df, ISIC2_REPORTS_DIR / "01_folder_structure.csv")
    save_csv(inventory_df, ISIC2_REPORTS_DIR / "02_image_inventory.csv")

    logger.info("Total images inventoried: %d", len(inventory_df))
    logger.info("Saved -> %s", ISIC2_REPORTS_DIR / "01_folder_structure.csv")
    logger.info("Saved -> %s", ISIC2_REPORTS_DIR / "02_image_inventory.csv")

    return inventory_df
